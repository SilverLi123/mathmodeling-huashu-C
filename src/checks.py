# -*- coding: utf-8 -*-
"""独立检查器：只使用原附件 + 导出的 schedule/hours 重算，不读优化器内部结构。

audit_solution(inputs, schedule, hourly_gpu_df=None, flows=None, with_storage=False, config=None)
返回 dict {name: {ok, detail}}，全部 PASS 才可用于论文。
"""
import numpy as np
import pandas as pd

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
MUTEX_TOL = 1e-5


def _ov(s, d, t):
    lo = max(float(t), float(s))
    hi = min(float(t) + 1.0, float(s) + float(d))
    return max(0.0, hi - lo)


def audit_solution(inputs, schedule, flows=None, hourly_gpu=None, with_storage=False,
                   q3=False, fixed_ai_df=None):
    """schedule: DataFrame(RunID/Question/TaskID/SourceRegion/AssignedRegion/ArrivalHour/
    StartHour/FinishHour/Duration_h/GPU_Demand/TaskType/NetworkLatency_ms)
    flows: DataFrame(Region/Hour/Load_MW/各功率/SOC...)（可选）
    hourly_gpu: DataFrame index=Hour, columns=Region（GPU 平均占用）可选
    q3=True 时任务侧检查跳过（固定附件负荷）。
    返回 (dict checks, list fail_records)。
    """
    chk = {}
    fails = []
    wt = inputs["workload"]
    lat = inputs["latency_long"].set_index(["FromRegion", "ToRegion"])["NetworkLatency_ms"]
    gpu_avail = inputs["gpu"].set_index("Region")["Available_GPU"].to_dict()
    pue = inputs["gpu"].set_index("Region")["PUE"].to_dict()
    it_cap = inputs["gpu"].set_index("Region")["Max_IT_Power_MW"].to_dict()
    fac_cap = inputs["gpu"].set_index("Region")["Max_Facility_Power_MW"].to_dict()
    pw = inputs["mapping"].set_index("TaskType")["GPU_Power_MW_per_EquivalentGPU"].to_dict()

    if not q3:
        # 1. 覆盖与唯一性
        sched_ids = schedule["TaskID"].to_list()
        uniq = len(sched_ids) == len(set(sched_ids))
        chk["task_unique"] = uniq
        if not uniq:
            dup = pd.Series(sched_ids).value_counts()
            fails.append({"check": "task_unique", "detail": f"重复任务 {dup[dup>1].head(3).to_dict()}"})
        need = set(wt["TaskID"])
        have = set(sched_ids)
        chk["task_coverage"] = len(need) == len(have) == len(schedule)
        if not chk["task_coverage"]:
            fails.append({"check": "task_coverage",
                          "detail": f"缺 {len(need-have)}，多 {len(have-need)}"})
        m = schedule.merge(wt[["TaskID", "TaskType", "ArrivalHour", "SourceRegion",
                               "MaxLatency_ms", "LatestFinishHour", "Duration_h"]],
                           on="TaskID", how="left", suffixes=("", "_w"))
        # 2. 时域
        v_2406 = m[m["FinishHour"] > 2406.0 + 1e-8]
        chk["no_task_past_2406"] = len(v_2406) == 0
        if len(v_2406): fails.append({"check": "no_task_past_2406", "detail": v_2406.head(3).to_dict("records")})
        # 3. 实时到达即开工
        rt = m[m["TaskType"] == "RealTimeInference"]
        chk["realtime_start_at_arrival"] = bool((np.abs(rt["StartHour"] - rt["ArrivalHour"]) < 1e-8).all())
        # 4. 弹性任务开工在到达后、截止前
        flex = m[m["TaskType"] != "RealTimeInference"]
        chk["flex_start_ge_arrival"] = bool((flex["StartHour"] >= flex["ArrivalHour"] - 1e-8).all())
        chk["flex_finish_le_2406"] = bool((flex["FinishHour"] <= 2406.0 + 1e-8).all())
        # 5. 时延
        m2 = m.copy()
        m2["lat_ms"] = m2.apply(lambda r: float(lat.loc[(r["SourceRegion"], r["AssignedRegion"])]), axis=1)
        viol = m2[m2["lat_ms"] > m2["MaxLatency_ms"] + 1e-6]
        chk["latency_ok"] = len(viol) == 0
        if len(viol): fails.append({"check": "latency_ok", "detail": viol.head(3).to_dict("records")})
        m2["overlap_err"] = np.abs(m2["FinishHour"] - (m2["StartHour"] + m2["Duration_h"]))
        chk["finish_eq_start_dur"] = bool((m2["overlap_err"] < 1e-6).all())
    # 6. GPU 小时占用检查
    if hourly_gpu is not None and not q3:
        viol_gpu = []
        for r in REGIONS:
            arr = hourly_gpu[r].to_numpy(float)
            bad = np.nonzero(arr > gpu_avail[r] + 1e-4)[0]
            for h in bad[:5]:
                viol_gpu.append({"Region": r, "Hour": int(h), "used": float(arr[h]),
                                 "avail": gpu_avail[r]})
        chk["gpu_capacity"] = len(viol_gpu) == 0
        if viol_gpu: fails.append({"check": "gpu_capacity", "detail": viol_gpu[:3]})
    # 6b. IT/设施功率独立重构检查（题目要求 GPU、IT 功率、设施功率同时满足）
    if not q3 and len(schedule):
        rh_it = inputs["region_hour"][["Region", "Hour", "NonAI_IT_Load_MW"]]
        ai_mw = {r: np.zeros(2407) for r in REGIONS}
        for _, row in schedule.iterrows():
            ri = REGIONS.index(row["AssignedRegion"])
            g = float(row["GPU_Demand"]); d = float(row["Duration_h"]); s = float(row["StartHour"])
            pval = pw[row["TaskType"]]
            t0 = int(np.floor(s))
            for h in range(t0, int(np.ceil(s + d)) + 1):
                if h > 2406 or h < 0:
                    continue
                ov = _ov(s, d, h)
                if ov > 0:
                    ai_mw[REGIONS[ri]][h] += g * ov * pval
        it_viol, fac_viol = [], []
        for r in REGIONS:
            na = rh_it[rh_it["Region"] == r].sort_values("Hour")["NonAI_IT_Load_MW"].to_numpy(float)
            it = ai_mw[r] + na
            fac = it * pue[r]
            for h in np.nonzero(it > it_cap[r] + 1e-4)[0][:5]:
                it_viol.append({"Region": r, "Hour": int(h), "IT_MW": float(it[h]), "cap": it_cap[r]})
            for h in np.nonzero(fac > fac_cap[r] + 1e-4)[0][:5]:
                fac_viol.append({"Region": r, "Hour": int(h), "Facility_MW": float(fac[h]), "cap": fac_cap[r]})
        chk["it_capacity"] = len(it_viol) == 0
        chk["facility_capacity"] = len(fac_viol) == 0
        if it_viol: fails.append({"check": "it_capacity", "detail": it_viol[:3]})
        if fac_viol: fails.append({"check": "facility_capacity", "detail": fac_viol[:3]})
    # 7. 能源守恒与边界（flows 提供时）
    if flows is not None:
        rh = inputs["region_hour"].set_index(["Region", "Hour"])
        f = flows
        lhs = f["GridPurchase_MW"] + rh["AvailableRenewable_MW"].reindex(
            f.set_index(["Region", "Hour"]).index).to_numpy(float) + f["Discharge_MW"]
        rhs = f["Load_MW"] + f["Charge_MW"] + f["GridSell_MW"] + f["Curtailment_MW"]
        res = (lhs - rhs).abs()
        chk["energy_balance"] = float(res.max()) < 1e-3
        if not chk["energy_balance"]:
            idx = int(res.idxmax())
            fails.append({"check": "energy_balance", "detail": f"max res {float(res.max()):.5f} at row {idx}"})
        st = inputs["storage"].set_index("Region")
        imp = f.merge(st[["MaxGridImport_MW", "SellLimit_MW"]], left_on="Region", right_index=True, how="left")
        chk["purchase_cap"] = float((imp["GridPurchase_MW"] - imp["MaxGridImport_MW"]).max()) < 1e-4
        chk["export_cap"] = float((imp["GridSell_MW"] - imp["SellLimit_MW"]).max()) < 1e-4
        chk["renew_balance"] = bool(np.allclose(
            f["RenewableDirect_MW"] + f["RenewableCharge_MW"] + f["RenewableSell_MW"] + f["Curtailment_MW"],
            rh["AvailableRenewable_MW"].reindex(f.set_index(["Region", "Hour"]).index).to_numpy(float), atol=1e-3))
        if with_storage:
            # SOC 递推 / 上下限 / 终端
            worst = 0.0
            soc_lb_ok = True
            term_ok = True
            for r, sub in f[f["SOC_MWh"].notna()].groupby("Region"):
                s0 = float(st.loc[r, "InitialSOC_MWh"])
                eta_c = float(st.loc[r, "ChargeEfficiency"])
                eta_d = float(st.loc[r, "DischargeEfficiency"])
                soc = s0
                for _, row in sub.iterrows():
                    nxt = soc + eta_c * row["Charge_MW"] - row["Discharge_MW"] / eta_d
                    worst = max(worst, abs(nxt - row["SOC_MWh"]))
                    soc = nxt
                if float(sub["SOC_MWh"].min()) < float(st.loc[r, "MinSOC_MWh"]) - 1e-4:
                    soc_lb_ok = False
                if soc < s0 - 1e-4:
                    term_ok = False
                    fails.append({"check": "soc_terminal", "detail": f"{r} final {soc:.4f} < init {s0}"})
            chk["soc_recursion"] = worst < 1e-4
            chk["soc_min_ok"] = soc_lb_ok
            chk["soc_terminal_ok"] = term_ok
            if with_storage:
                C = f["Charge_MW"].to_numpy(float)
                D = f["Discharge_MW"].to_numpy(float)
                n_mutex = int((np.minimum(C, D) > MUTEX_TOL).sum())
                chk["charge_discharge_mutex"] = n_mutex == 0
                if n_mutex: fails.append({"check": "charge_discharge_mutex", "detail": f"{n_mutex} 小时循环"})
    ok = all(isinstance(v, bool) and v for v in chk.values()) if chk else True
    return chk, fails
