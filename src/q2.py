# -*- coding: utf-8 -*-
"""Q2 无储能碳感知任务调度。

方法：以可行基线计算的边际成本面 λ_rt（三段 MC：弃电=0 / 外送=π_sell / 购电=π_buy），
对弹性任务按 Score=g·p_k·PUE·Σ_h α_ist·λ_rt 做候选内代价引导重指派（前缀和 O(1) 计分
+ 弃电区段定向候选 + 容量约束），多 pass 重算 λ；完整能源结算复算真实成本/碳。实时任务
到达即开工。无储能下成本与碳为权衡（碳随迁移时延预算先升后落）；产出成本-时延折中（迁移时延预算）。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.io_data import REGIONS
from src.task_model import TaskScheduler, feasible_regions, _ov
from src.energy_model import _region_params, no_storage_mc, settle_no_storage, eval_flows, make_facility_load
from src.metrics import schedule_metrics
from src.checks import audit_solution


def _mc_surface(inputs, load):
    cols = {}
    for r in REGIONS:
        p = _region_params(inputs, r)
        L = load[r].to_numpy(float)
        mc = no_storage_mc(L, p["ren"], p["sell"], p["buy"], p["export_cap"])
        cols[r] = mc
    return pd.DataFrame(cols, index=np.arange(2407))


def _score(P_r, MC_r, s, g, pk, pue, d):
    s = int(s)
    k = int(np.floor(d))
    frac = d - k
    if k >= 1:
        seg = float(P_r[s + k] - P_r[s])
        if frac > 0 and s + k <= 2406:
            seg += frac * float(MC_r[s + k])
    else:
        seg = d * float(MC_r[s]) if s <= 2406 else 0.0
    return g * pk * pue * seg


def _zero_intervals(mc_arr):
    z = np.abs(np.asarray(mc_arr)) < 1e-9
    ivs, i = [], 0
    while i < len(z):
        if z[i]:
            j = i
            while j < len(z) and z[j]:
                j += 1
            ivs.append((i, j))
            i = j
        else:
            i += 1
    return ivs


def min_cost_schedule(inputs, wl, latency_budget=150.0, passes=2, init_schedule=None):
    """从可行基线（默认本地-到达；也可由 init_schedule 提供 incumbent）出发，对弹性任务做
    代价引导改进：仅当目标槽可行且 Score 严格低于当前占位时移动。保证可行性恒成立且
    结果成本不劣于起始解（incumbent 单调）。返回 (schedule, None)；schedule 含 NetworkLatency_ms。
    """
    if init_schedule is not None:
        base = init_schedule
    else:
        base, err = TaskScheduler(inputs).construct(wl)
        if err is not None:
            return None, err
    ts = TaskScheduler(inputs)
    ts.reset()
    for _, r in base.iterrows():
        ts.add(r["TaskID"], r["AssignedRegion"], r["StartHour"],
               float(r["GPU_Demand"]), float(r["Duration_h"]))
    pw = dict(zip(inputs["mapping"]["TaskType"],
                  inputs["mapping"]["GPU_Power_MW_per_EquivalentGPU"].astype(float)))
    pue = inputs["gpu"].set_index("Region")["PUE"].to_dict()
    wl_idx = wl.set_index("TaskID")
    feas_map = {}
    for (src, k), sub in wl.groupby(["SourceRegion", "TaskType"]):
        maxlat = float(sub["MaxLatency_ms"].iloc[0])
        feas_map[(src, k)] = [r for r, ms in feasible_regions(inputs, src, maxlat)
                              if ms <= latency_budget + 1e-9]
    order = wl[wl["TaskType"] != "RealTimeInference"] \
        .assign(work=lambda d: d["GPU_Demand"] * d["Duration_h"]) \
        .sort_values("work", ascending=False)["TaskID"].tolist()
    latmat = inputs["latency"]
    for _pass in range(passes):
        mc = _mc_surface(inputs, make_facility_load(inputs, ts.build_ai_mw_sched(inputs)))
        P = {r: np.concatenate([[0.0], np.cumsum(mc[r].to_numpy(float))]) for r in REGIONS}
        MC = {r: mc[r].to_numpy(float) for r in REGIONS}
        ZIV = {r: _zero_intervals(MC[r]) for r in REGIONS}
        for tid in order:
            t = wl_idx.loc[tid]
            regs = feas_map[(t["SourceRegion"], t["TaskType"])]
            g = float(t["GPU_Demand"]); d = float(t["Duration_h"])
            a = float(t["ArrivalHour"]); latest = 2406 - np.ceil(d)
            pk = pw[t["TaskType"]]
            cur_r, cur_s = ts.placements[tid]
            cur_score = _score(P[cur_r], MC[cur_r], cur_s, g, pk, pue[cur_r], d)
            best = None
            for r in regs:
                cand = {int(a), int(latest), int(cur_s if cur_r == r else a)}
                for s in range(int(a), min(int(a) + 5, int(latest) + 1)):
                    cand.add(s)
                zivs = ZIV[r]
                for idx in range(0, len(zivs), max(1, len(zivs) // 4)):
                    lo, hi = zivs[idx]
                    s0 = max(int(a), lo)
                    if s0 <= int(latest) and s0 < hi:
                        cand.add(s0)
                        cand.add(min(hi - 1, int(latest)))
                for s in sorted(cand):
                    if s < a - 1e-9 or s > latest + 1e-9:
                        continue
                    sc = _score(P[r], MC[r], s, g, pk, pue[r], d)
                    if sc > cur_score - 1e-9:
                        continue
                    if r == cur_r and int(s) == int(cur_s):
                        continue
                    if not ts.fits_after_move(cur_r, cur_s, r, float(s), g, d, pk):
                        continue
                    if best is None or sc < best[0] - 1e-12:
                        best = (sc, r, float(s))
            if best is not None:
                _, r, s = best
                ts.remove(tid, cur_r, cur_s, g, d)
                ts.add(tid, r, s, g, d)
    rows = []
    for tid, (r, s) in ts.placements.items():
        t = wl_idx.loc[tid]
        rows.append({"TaskID": tid, "SourceRegion": t["SourceRegion"], "AssignedRegion": r,
                     "ArrivalHour": t["ArrivalHour"], "StartHour": s,
                     "FinishHour": s + float(t["Duration_h"]), "Duration_h": float(t["Duration_h"]),
                     "GPU_Demand": float(t["GPU_Demand"]), "TaskType": t["TaskType"],
                     "NetworkLatency_ms": float(latmat.loc[t["SourceRegion"], r])})
    return pd.DataFrame(rows).sort_values("TaskID"), None


def run_q2(inputs, cfg, fixed_schedule=None, out_dir="results/runs/base/q2", latency_budgets=None, passes=2):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    wl = inputs["workload"]
    budgets = sorted(latency_budgets or [20.0, 80.0, 150.0])
    results = {}
    prev = None
    for lb in budgets:
        sched, err = min_cost_schedule(inputs, wl, latency_budget=lb, passes=passes,
                                       init_schedule=prev)
        if err is not None:
            results[lb] = {"status": "INFEASIBLE", "detail": str(err)}
            continue
        ts = TaskScheduler(inputs); ts.reset()
        for _, r in sched.iterrows():
            ts.add(r["TaskID"], r["AssignedRegion"], r["StartHour"], float(r["GPU_Demand"]), float(r["Duration_h"]))
        load = make_facility_load(inputs, ts.build_ai_mw_sched(inputs))
        flows = settle_no_storage(inputs, load)
        en = eval_flows(inputs, flows) if flows is not None else {"cost": np.nan, "carbon": np.nan, "ren_util": np.nan}
        hg = ts.build_hourly_gpu()
        m = schedule_metrics(sched, inputs, flows=flows, hourly_gpu=hg)
        chk, fails = audit_solution(inputs, sched, flows=flows, hourly_gpu=hg, with_storage=False)
        results[lb] = {"status": "OK", "latency_budget_ms": lb, "energy": en,
                       "sched_metrics": m, "checks": chk, "check_fails": len(fails)}
        sched.to_csv(out / f"schedule_q2_lb{int(lb)}.csv", index=False)
        if flows is not None:
            flows.to_csv(out / f"hourly_energy_q2_lb{int(lb)}.csv", index=False)
        prev = sched
    N_res = None
    if fixed_schedule is not None:
        ts = TaskScheduler(inputs); ts.reset()
        for _, r in fixed_schedule.iterrows():
            ts.add(r["TaskID"], r["AssignedRegion"], r["StartHour"], float(r["GPU_Demand"]), float(r["Duration_h"]))
        flows = settle_no_storage(inputs, make_facility_load(inputs, ts.build_ai_mw_sched(inputs)))
        N_res = {"energy": eval_flows(inputs, flows),
                 "sched_metrics": schedule_metrics(fixed_schedule, inputs, flows=flows,
                                                   hourly_gpu=ts.build_hourly_gpu())}
    (out / "metrics_q2.json").write_text(json.dumps({"T": results, "N": N_res},
                                                    ensure_ascii=False, indent=1, default=str))
    return {"T": results, "N": N_res}
def run_q2_carbon_envelope(q2_result, out_dir="results/runs/base/q2/carbon_pareto"):
    """无储能 Cost-Carbon 前沿：从 Q2 时延预算方案族 {N, T20, T80, T150} 收集
    (cost, carbon) 点，取非支配下包络；ε-约束解读：给定碳预算 ε，最低成本 =
    min{cost_i : carbon_i ≤ ε}（可行方案族下包络）。"""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pts = []
    N = q2_result.get("N")
    if isinstance(N, dict) and isinstance(N.get("energy"), dict):
        pts.append({"label": "N(不迁移)", "cost": float(N["energy"]["cost"]),
                    "carbon": float(N["energy"]["carbon"])})
    for lb, v in q2_result["T"].items():
        if isinstance(v, dict) and v.get("status") == "OK":
            pts.append({"label": f"T{int(float(lb))}", "cost": float(v["energy"]["cost"]),
                        "carbon": float(v["energy"]["carbon"])})
    ok = sorted(pts, key=lambda p: p["cost"])
    pareto, best = [], float("inf")
    for p in ok:
        if p["carbon"] < best - 1e-9:
            pareto.append(p); best = p["carbon"]
    eps = [10.0, 50.0, 100.0, 300.0, 500.0, 800.0, 1000.0]
    envelope = []
    for e in eps:
        feas = [p for p in pts if p["carbon"] <= e + 1e-9]
        if feas:
            arg = min(feas, key=lambda p: p["cost"])
            envelope.append({"epsilon": e, "status": "OK", "cost": arg["cost"], "argmin": arg["label"]})
        else:
            envelope.append({"epsilon": e, "status": "INFEASIBLE", "cost": None, "argmin": None})
    payload = {"points": pts, "pareto": pareto, "envelope": envelope}
    (out / "carbon_envelope_q2.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    return payload

