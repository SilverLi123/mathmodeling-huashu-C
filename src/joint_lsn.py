# -*- coding: utf-8 -*-
"""任务+储能联合优化的存储感知局部搜索（LNS，批量回退 deconstruction 变体）。

关键事实（实测）：储能最优成本对个别任务的迁移几乎不敏感（单任务回退 Δ≈0），
只有成批回退若干任务才显现成本变化，说明储能显著覆盖了任务的时序/分区套利空间。
故采用"批量回退"邻域：对已迁移任务按工作量降序，按 K 递增批量回退到本地基线，
逐次对储能子问题(LP-first 最优)真实重解，仅接受总成本下降，最终逼近联合最优。

返回 dict(best_cost, best_incumbent, log, n_reverted)。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.io_data import REGIONS
from src.task_model import TaskScheduler
from src.energy_model import (make_facility_load, solve_region_storage_lp_first,
                              eval_flows)


def _storage_cost(inputs, ts):
    ai = ts.build_ai_mw_sched(inputs)
    load = make_facility_load(inputs, ai)
    allf = []
    for r in REGIONS:
        res = solve_region_storage_lp_first(inputs, r, load[r].to_numpy(float))
        if res.get("flows") is not None:
            allf.append(res["flows"])
    flows = pd.concat(allf, ignore_index=True) if allf else None
    if flows is None:
        return np.nan
    return eval_flows(inputs, flows)["cost"]


def _schedule_from_ts(inputs, ts):
    wl = inputs["workload"].set_index("TaskID")
    lat = inputs["latency"]
    rows = []
    for tid, (r, s) in ts.placements.items():
        t = wl.loc[tid]
        rows.append({"TaskID": tid, "SourceRegion": t["SourceRegion"], "AssignedRegion": r,
                     "ArrivalHour": t["ArrivalHour"], "StartHour": s,
                     "FinishHour": s + float(t["Duration_h"]), "Duration_h": float(t["Duration_h"]),
                     "GPU_Demand": float(t["GPU_Demand"]), "TaskType": t["TaskType"],
                     "NetworkLatency_ms": float(lat.loc[t["SourceRegion"], r])})
    return pd.DataFrame(rows).sort_values("TaskID")


def joint_storage_lsn(inputs, n_schedule, t_schedule, top_k=None):
    """批量回退 LNS：在候选{本地N、迁移T、批量回退中间态}上求储能子问题最优，取 min。"""
    local = {r["TaskID"]: (r["AssignedRegion"], float(r["StartHour"]))
             for _, r in n_schedule.iterrows()}
    nn = n_schedule.set_index("TaskID")
    pw = dict(zip(inputs["mapping"]["TaskType"],
                  inputs["mapping"]["GPU_Power_MW_per_EquivalentGPU"].astype(float)))
    # 偏离本地基线的任务：区域改变 或 开工时间改变（两者都是 Q2 迁移的作用）
    migrated = []
    for _, r in t_schedule.iterrows():
        tid = r["TaskID"]
        lr = nn.loc[tid, "AssignedRegion"]
        ls = float(nn.loc[tid, "StartHour"])
        if r["AssignedRegion"] != lr or float(r["StartHour"]) != ls:
            g = float(r["GPU_Demand"]); d = float(r["Duration_h"])
            migrated.append({"tid": tid, "cur_r": r["AssignedRegion"], "cur_s": float(r["StartHour"]),
                             "loc_r": lr, "loc_s": ls, "g": g, "d": d,
                             "p": pw[nn.loc[tid, "TaskType"]], "work": g * d})
    migrated.sort(key=lambda x: -x["work"])
    Nm = len(migrated)

    def build_from(schedule):
        """直接从排程构建任务状态（不经回退，用于本地 N / 迁移 T 两个基线）。"""
        ts = TaskScheduler(inputs); ts.reset()
        for _, r in schedule.iterrows():
            ts.add(r["TaskID"], r["AssignedRegion"], float(r["StartHour"]),
                   float(r["GPU_Demand"]), float(r["Duration_h"]))
        return ts

    def build(pick):  # pick: 选择哪些偏离任务回退到本地
        ts = build_from(t_schedule)
        for idx in pick:
            m = migrated[idx]
            if ts.fits_after_move(m["cur_r"], m["cur_s"], m["loc_r"], m["loc_s"], m["g"], m["d"], m["p"]):
                ts.remove(m["tid"], m["cur_r"], m["cur_s"], m["g"], m["d"])
                ts.add(m["tid"], m["loc_r"], m["loc_s"], m["g"], m["d"])
        return ts

    if top_k is None:
        top_k = min(1000, Nm)
    ks = sorted(set([0, 5, 10, 25, 50, 100, 250, 500, 1000, Nm]) & {k for k in [0, 5, 10, 25, 50, 100, 250, 500, 1000, Nm] if k <= top_k})
    if Nm not in ks:
        ks = ks + [Nm]
    log = []
    best = None
    best_ts = None
    best_tag = None
    for K in ks:
        pick = list(range(K))
        ts = build(pick)
        c = _storage_cost(inputs, ts)
        log.append({"K_reverted": K, "cost": c, "note": "revert_topK" if K > 0 else "two_stage_T"})
        if best is None or c < best - 1e-6:
            best = c; best_ts = ts; best_tag = f"revert_top{K}"
    # 本地 N 与迁移 T 直接由原排程构建（与 F01 / 两阶段 F11 同口径，可互相对账）
    tsN = build_from(n_schedule)
    cN = _storage_cost(inputs, tsN)
    log.append({"K_reverted": Nm, "cost": cN, "note": "local_N_direct"})
    if cN < best - 1e-6:
        best = cN; best_ts = tsN; best_tag = "local_N"
    tsT = build_from(t_schedule)
    cT = _storage_cost(inputs, tsT)
    log.append({"K_reverted": 0, "cost": cT, "note": "two_stage_T_direct"})
    if cT < best - 1e-6:
        best = cT; best_ts = tsT; best_tag = "two_stage_T"
    best_sched_df = _schedule_from_ts(inputs, best_ts) if best_ts is not None else None
    return {"best_cost": best, "best_incumbent": best_tag, "log": log,
            "n_migrated": Nm, "n_reverted": Nm if best_tag in ("local_N",) else 0,
            "local_N_cost": cN, "two_stage_T_cost": cT,
            "best_schedule": best_sched_df}


def run_joint_lsn(inputs, cfg, n_schedule=None, t_schedule=None,
                  out_dir="results/runs/base/q4", top_k=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    res = joint_storage_lsn(inputs, n_schedule, t_schedule, top_k=top_k)
    res = dict(res)
    if res.get("best_schedule") is not None:
        bs = res["best_schedule"]
        bs.to_csv(out / "joint_lsn_best_schedule.csv", index=False)
        # 复算并导出 best 的储能能流，供独立验收
        ts = TaskScheduler(inputs); ts.reset()
        for _, r in bs.iterrows():
            ts.add(r["TaskID"], r["AssignedRegion"], float(r["StartHour"]),
                   float(r["GPU_Demand"]), float(r["Duration_h"]))
        ai = ts.build_ai_mw_sched(inputs)
        load = make_facility_load(inputs, ai)
        allf = []
        for r in REGIONS:
            rr = solve_region_storage_lp_first(inputs, r, load[r].to_numpy(float))
            if rr.get("flows") is not None:
                allf.append(rr["flows"])
        if allf:
            fl = pd.concat(allf, ignore_index=True)
            fl.to_csv(out / "joint_lsn_best_flows.csv", index=False)
            res["best_verified_cost"] = float(eval_flows(inputs, fl)["cost"])
    del res["best_schedule"]
    (out / "joint_lsn.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return res