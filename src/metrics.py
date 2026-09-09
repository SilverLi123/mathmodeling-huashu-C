# -*- coding: utf-8 -*-
"""统一指标：成本/碳/利用率/时延/等待/峰值/爬坡/储能吞吐。"""
import numpy as np
import pandas as pd

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]


def schedule_metrics(schedule: pd.DataFrame, inputs: dict, flows: pd.DataFrame = None,
                     hourly_gpu: pd.DataFrame = None) -> dict:
    """任务侧指标（与能源无关部分）。schedule 需含 TaskID/StartHour/FinishHour/Duration_h/
    AssignedRegion/SourceRegion/ArrivalHour/TaskType/NetworkLatency_ms。
    返回 dict：n, mean_wait_h, wait_by_type(mean/p95), slowdown_by_type, mean_latency_ms,
    latency_by_type, peak_gpu_util(region), 2406_ai_load_ok 等。"""
    s = schedule
    s = s.assign(wait_h=s["StartHour"] - s["ArrivalHour"],
                 lat_ms=s["NetworkLatency_ms"].astype(float),
                 slow=(s["FinishHour"] - s["ArrivalHour"]) / s["Duration_h"].clip(lower=1e-9))
    out = {"n_tasks": int(len(s)),
           "mean_wait_h": float(s["wait_h"].mean()),
           "p95_wait_h": float(np.percentile(s["wait_h"], 95)),
           "mean_latency_ms": float(s["lat_ms"].mean()),
           "sum_wait_h": float(s["wait_h"].sum())}
    w = {}
    sl = {}
    la = {}
    for k, sub in s.groupby("TaskType"):
        w[k] = {"mean_h": float(sub["wait_h"].mean()), "p95_h": float(np.percentile(sub["wait_h"], 95))}
        sl[k] = {"mean": float(sub["slow"].mean()), "p95": float(np.percentile(sub["slow"], 95))}
        la[k] = {"mean_ms": float(sub["lat_ms"].mean())}
    out["wait_by_type"] = w
    out["slowdown_by_type"] = sl
    out["latency_by_type"] = la
    out["migration"] = {
        "n_migrated": int((s["AssignedRegion"] != s["SourceRegion"]).sum()),
        "share": float((s["AssignedRegion"] != s["SourceRegion"]).mean()),
    }
    # 迁移矩阵
    mm = s.groupby(["SourceRegion", "AssignedRegion"]).size().unstack(fill_value=0)
    out["migration_matrix"] = mm.reindex(index=REGIONS, columns=REGIONS, fill_value=0).to_dict()
    if flows is not None:
        f = flows
        net = (f["GridPurchase_MW"] - f["GridSell_MW"])
        g = f.groupby("Region")
        out["peak_net_import_MW"] = {r: float(g.get_group(r)["GridPurchase_MW"].max()) for r in REGIONS}
        # 分区净购电峰值按 B（含充）口径与 max_t(B−P_sell) 双口径
        nf = f.assign(net=f["GridPurchase_MW"] - f["GridSell_MW"])
        out["peak_netpurchase_MW"] = {r: float(nf[nf.Region == r]["GridPurchase_MW"].max()) for r in REGIONS}
        ramps = {}
        for r in REGIONS:
            sub = nf[nf.Region == r].sort_values("Hour")
            v = sub["net"].to_numpy(float)
            ramps[r] = float(np.abs(np.diff(v)).mean()) if len(v) > 1 else 0.0
        out["mean_abs_ramp_MW"] = ramps
    if hourly_gpu is not None:
        avail = inputs["gpu"].set_index("Region")["Available_GPU"].to_dict()
        out["peak_gpu_util"] = {r: float(hourly_gpu[r].max() / avail[r]) for r in REGIONS if avail.get(r)}
    return out


def storage_metrics(flows: pd.DataFrame, inputs: dict) -> dict:
    f = flows[flows["SOC_MWh"].notna()]
    if len(f) == 0:
        return {}
    st = inputs["storage"].set_index("Region")
    out = {}
    for r, sub in f.groupby("Region"):
        soc0 = float(st.loc[r, "InitialSOC_MWh"])
        C = sub["Charge_MW"].sum()
        D = sub["Discharge_MW"].sum()
        eta_c = float(st.loc[r, "ChargeEfficiency"])
        eta_d = float(st.loc[r, "DischargeEfficiency"])
        loss = (1 - eta_c) * C + (1 / eta_d - 1) * D
        out[r] = {"charge_MWh": float(C), "discharge_MWh": float(D),
                  "loss_MWh": float(loss),
                  "soc_final": float(sub["SOC_MWh"].iloc[-1]),
                  "soc_min": float(sub["SOC_MWh"].min()),
                  "soc_terminal_ok": float(sub["SOC_MWh"].iloc[-1]) >= soc0 - 1e-6}
    return out
