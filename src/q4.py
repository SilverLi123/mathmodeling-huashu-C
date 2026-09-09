# -*- coding: utf-8 -*-
"""Q4：任务+储能联合优化（四格 F00/F10/F01/F11 + 交互 + 场景）。

同合同重解：四格均基于同一可行任务族（Q4_feasible_tasks）与同一能源边界。
  F00 = 固定任务(基线N) + 无储能；F10 = 任务优化(T) + 无储能；
  F01 = 固定任务(N负荷) + 储能；F11 = 任务优化(T负荷) + 储能（任务先优化→固定新负荷重求储能）。
交互 I = F10+F01−F11−F00。
说明：任务侧为启发式可行策略，储能侧为 LP-first/MILP 的（子问题）最优；无有效全问题下界，
故不报告认证交互区间，仅报告可行策略点估计（符合合同）。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.io_data import REGIONS
from src.task_model import TaskScheduler
from src.energy_model import (settle_no_storage, eval_flows, make_facility_load,
                              solve_region_storage_lp_first)


def _load_to_ai(inputs, schedule_or_none):
    ts = TaskScheduler(inputs)
    ts.reset()
    if schedule_or_none is not None:
        for _, r in schedule_or_none.iterrows():
            ts.add(r["TaskID"], r["AssignedRegion"], r["StartHour"],
                   float(r["GPU_Demand"]), float(r["Duration_h"]))
    return ts.build_ai_mw_sched(inputs)


def _settle(inputs, ai_mw, with_storage=False):
    load = make_facility_load(inputs, ai_mw)
    if not with_storage:
        flows = settle_no_storage(inputs, load)
        return eval_flows(inputs, flows) if flows is not None else {"cost": np.nan, "carbon": np.nan, "ren_util": np.nan}, flows, load
    allf = []
    for r in REGIONS:
        res = solve_region_storage_lp_first(inputs, r, load[r].to_numpy(float))
        if res.get("flows") is not None:
            allf.append(res["flows"])
    flows = pd.concat(allf, ignore_index=True) if allf else None
    return eval_flows(inputs, flows) if flows is not None else {"cost": np.nan, "carbon": np.nan, "ren_util": np.nan}, flows, load


def run_q4(inputs, cfg, n_schedule=None, t_schedule=None, out_dir="results/runs/base/q4"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    res = {}
    # F00：固定任务(本地调度 N) + 无储能
    aiN = _load_to_ai(inputs, n_schedule)
    en00, fl00, lo00 = _settle(inputs, aiN, with_storage=False)
    res["F00"] = {"label": "固定任务+无储能", "energy": en00}
    res["F00"]["N"] = {"cost": en00["cost"], "carbon": en00["carbon"]}  # 供跨表引用
    # F10：任务优化 + 无储能 —— 保留"不迁移"incumbent，强制 F10≤F00（任务灵活性收益非负）
    aiT = _load_to_ai(inputs, t_schedule) if t_schedule is not None else aiN
    en10g, fl10g, lo10g = _settle(inputs, aiT, with_storage=False)
    if en10g["cost"] <= en00["cost"] + 1e-6:
        en10, fl10, lo10, mig_used = en10g, fl10g, lo10g, True
    else:
        en10, fl10, lo10, mig_used = en00, fl00, lo00, False
    res["F10"] = {"label": "任务优化+无储能", "energy": en10, "migration_used": mig_used}
    # F01：固定任务 + 储能（公共基线 aiN，非 Q3 附件负荷；储能子问题 LP-first 最优）
    en01, fl01, lo01 = _settle(inputs, aiN, with_storage=True)
    res["F01"] = {"label": "固定任务+储能", "energy": en01}
    # F11：联合 —— 两阶段(迁移+储能)与全部 incumbent 取最优，强制 F11≤min(F01,F10,F00)
    en11g, fl11g, lo11g = _settle(inputs, aiT, with_storage=True) if t_schedule is not None else (en01, fl01, lo01)
    cand = [(label, en, fl, lo) for label, en, fl, lo in [
        ("两阶段(迁移+储能)", en11g, fl11g, lo11g),
        ("F01(固定任务+储能)", en01, fl01, lo01),
        ("F10(任务优化+无储能)", en10, fl10, lo10)]
        if np.isfinite(en["cost"])]
    if not cand:
        branch, en11, fl11, lo11 = "F01(固定任务+储能)", en01, fl01, lo01
    else:
        branch, en11, fl11, lo11 = min(cand, key=lambda c: c[1]["cost"])
    res["F11"] = {"label": "任务优化+储能", "energy": en11, "incumbent_source": branch}
    I = en10["cost"] + en01["cost"] - en11["cost"] - en00["cost"]
    res["interaction"] = {"I": I,
                          "note": "可行策略点估计，任务侧无有效全问题下界，不报告认证区间",
                          "formula": "F10+F01-F11-F00",
                          "F11_incumbent_source": branch}
    res["storage_saving_joint"] = {"fixed": en00["cost"] - en01["cost"],
                                   "task": en10["cost"] - en11["cost"]}
    res["task_saving"] = {"no_storage": en00["cost"] - en10["cost"],
                          "with_storage": en01["cost"] - en11["cost"]}
    res["F01_common_baseline"] = "F01 在 Q4 公共基线(本地调度 aiN)上重解储能，与 F00 同负荷体系（非 Q3 附件负荷）"
    if fl00 is not None: fl00.to_csv(out / "flows_F00.csv", index=False)
    if fl10 is not None: fl10.to_csv(out / "flows_F10.csv", index=False)
    if fl01 is not None: fl01.to_csv(out / "flows_F01.csv", index=False)
    if fl11 is not None: fl11.to_csv(out / "flows_F11.csv", index=False)
    (out / "metrics_q4.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return res