# -*- coding: utf-8 -*-
"""场景分析：新能源规模缩减 ρ、电价机制扰动、短时新能源出力缺口。

对同一任务基线族重算 F00/F10/F01/F11（固定任务族=基线下 N 与 T 负荷、储能子问题重解）。
ρ 扫描用于粗扫与局部细扫任务迁移/储能节省的变化，并报告策略转折区间（粗扫粒度）。
"""
import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from src.io_data import REGIONS
from src.q4 import run_q4, _settle, _load_to_ai


def scaled_inputs(inputs, ren_scale=1.0, sell_scale=1.0, buy_scale=1.0, outage=None):
    """返回 region_hour 经缩放的新 inputs 副本（浅复制其余部件）。"""
    ni = {k: v for k, v in inputs.items()}
    rh = inputs["region_hour"].copy()
    if ren_scale != 1.0:
        rh["AvailableRenewable_MW"] = rh["AvailableRenewable_MW"] * ren_scale
    if sell_scale != 1.0:
        rh["SellPrice_CNY_per_MWh"] = rh["SellPrice_CNY_per_MWh"] * sell_scale
    if buy_scale != 1.0:
        rh["ElectricityPrice_CNY_per_MWh"] = rh["ElectricityPrice_CNY_per_MWh"] * buy_scale
    if outage is not None:
        for (h0, h1), f in outage.items():
            mask = (rh["Hour"] >= h0) & (rh["Hour"] <= h1)
            rh.loc[mask, "AvailableRenewable_MW"] = rh.loc[mask, "AvailableRenewable_MW"] * f
    ni["region_hour"] = rh
    return ni


def scenario_point(inputs, n_schedule, t_schedule, ren_scale=1.0, sell_scale=1.0, buy_scale=1.0,
                   outage=None, reoptimize=True, latency_budget=150.0):
    """单情景四格：在缩放后的新能源/电价下，**重跑任务迁移 incumbent 链条（20→80→150ms）**，
    得到场景特定的 150ms 迁移排程（与基准 Q4 一致的方法），而非复用原场景排程。
    若迁移未找到可行解则 t_schedule=None（run_q4 回退"不迁移"，如实表达"未找到"）。"""
    ni = scaled_inputs(inputs, ren_scale, sell_scale, buy_scale, outage)
    t_scen = t_schedule
    t_status = "reused"
    if reoptimize:
        from src.q2 import min_cost_schedule
        wl = ni["workload"]
        prev = n_schedule
        for lb in [20.0, 80.0, latency_budget]:
            t_new, err = min_cost_schedule(ni, wl, latency_budget=lb, passes=2,
                                           init_schedule=prev)
            if err is not None:
                t_scen = None
                t_status = "no_feasible_T"
                break
            prev = t_new
        else:
            t_scen = prev
            t_status = "reoptimized"
    q = run_q4(ni, {}, n_schedule=n_schedule, t_schedule=t_scen,
               out_dir="results/runs/scenarios/_tmp")
    q["task_reoptimize_status"] = t_status
    return q


def run_scenarios(inputs, cfg, n_schedule=None, t_schedule=None, out_dir="results/runs/scenarios"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rhos = cfg["scenario_renewable_scale"]
    rows = []
    for rho in rhos:
        q = scenario_point(inputs, n_schedule, t_schedule, ren_scale=rho)
        rows.append({"scenario": f"ren_scale={rho}", "rho": rho,
                     "F00": q["F00"]["energy"]["cost"], "F10": q["F10"]["energy"]["cost"],
                     "F01": q["F01"]["energy"]["cost"], "F11": q["F11"]["energy"]["cost"],
                     "I": q["interaction"]["I"],
                     "task_saving": q["task_saving"]["no_storage"],
                     "storage_saving_fixed": q["storage_saving_joint"]["fixed"],
                     "storage_saving_task": q["storage_saving_joint"]["task"],
                     "F10_migration_used": bool(q["F10"].get("migration_used", True)),
                     "F11_incumbent_source": q["F11"].get("incumbent_source", ""),
                     "F00_carbon": q["F00"]["energy"]["carbon"],
                     "F11_carbon": q["F11"]["energy"]["carbon"]})
    sc_df = pd.DataFrame(rows)
    # 任务迁移增益：受 incumbent 保护恒非负（F00−F10≥0），报告其"衰减至 0"的区间（True→False）
    mig_used = sc_df["F10_migration_used"].astype(bool).to_numpy()
    task_decay = [None, None]
    for i in range(len(rhos) - 1):
        if mig_used[i] and not mig_used[i + 1]:
            task_decay = [float(rhos[i]), float(rhos[i + 1])]
            break
    # 储能节省：恒正（F01≤F00），报告其最小值位置而非符号翻转
    ssig = np.sign(sc_df["storage_saving_fixed"].to_numpy())
    stor_flip = [None, None]
    for i in range(len(rhos) - 1):
        if ssig[i] * ssig[i + 1] < 0:
            stor_flip = [float(rhos[i]), float(rhos[i + 1])]
            break
    trans = [{"name": "task_migration_gain_decays_F00-F10", "interval": task_decay,
              "note": "任务迁移独立增益(≥0)随新能源供给下降而衰减至 0"},
             {"name": "storage_saving_F00-F01", "interval": stor_flip,
              "note": "储能节省恒正(区间 None=未翻转)"}]
    # 新能源缺口场景
    outage = {tuple(cfg["scenario_low_outage"]["hours"]): cfg["scenario_low_outage"]["factor"]}
    q_low = scenario_point(inputs, n_schedule, t_schedule, outage=outage)
    # 电价场景
    q_price = scenario_point(inputs, n_schedule, t_schedule, sell_scale=0.7, buy_scale=1.3)
    special = {"low_renewable_outage": {k: (q_low[k]["energy"]["cost"] if isinstance(q_low[k], dict) and "energy" in q_low[k] else q_low[k])
                                         for k in ("F00", "F10", "F01", "F11")},
               "price_spread": {k: (q_price[k]["energy"]["cost"] if isinstance(q_price[k], dict) and "energy" in q_price[k] else q_price[k])
                                for k in ("F00", "F10", "F01", "F11")}}
    sc_df.to_csv(out / "scenario_renewable_scale.csv", index=False)
    result = {"renewable_scale": rows, "transition_intervals": trans, "special": special,
              "q_low_detail": q_low, "q_price_detail": q_price}
    (out / "scenarios.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    return result