# -*- coding: utf-8 -*-
"""主编排：T00-T08 全量求解链（数据→Q1→Q2→Q3→Q4→场景）。唯一复现命令入口。"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.io_data import load_inputs, _validate_anchors
from src.q1 import run_q1
from src.q2 import run_q2
from src.q3 import run_q3
from src.q4 import run_q4
from src.scenarios import run_scenarios


def main():
    t_start = time.time()
    root = Path(".")
    cfg = json.load(open("configs/base.json"))
    print("[T01] load inputs", flush=True)
    inputs = load_inputs(cfg["input_dir"])
    anchors = _validate_anchors(inputs)
    assert all(ok for ok, _ in anchors.values()), {k: v for k, (ok, v) in anchors.items() if not ok}
    print("[T01] anchors OK; hashes:", {k: v[:8] for k, v in inputs["hashes"].items()}, flush=True)

    print("[T04] Q1 forecast (ridge vs mean baseline)", flush=True)
    from src.forecast import run_forecast
    fc = run_forecast(inputs["workload"], cfg)
    Path("results/q1_forecast.json").write_text(json.dumps({
        "choice": fc["choice"],
        "metric": fc["metric"],
        "baseline_metric": fc["baseline_metric"],
    }, ensure_ascii=False, indent=1, default=str))
    print("  choice=", fc["choice"], "ridge=", fc["metric"],
          "baseline=", fc["baseline_metric"], flush=True)

    print("[T05] Q1 base schedule", flush=True)
    q1 = run_q1(inputs, cfg)
    print("  Q1", q1["status"], "energy", q1.get("energy"), "fails", q1["check_fails"], flush=True)

    n_sched = pd.read_csv("results/runs/base/q1/schedule_q1.csv")

    print("[T06] Q2 no-storage scheduling", flush=True)
    q2 = run_q2(inputs, cfg, fixed_schedule=n_sched, passes=2)
    t_sched = pd.read_csv(f"results/runs/base/q2/schedule_q2_lb150.csv")

    print("[T06b] Q2 no-storage carbon envelope (epsilon-constraint)", flush=True)
    from src.q2 import run_q2_carbon_envelope
    cpar = run_q2_carbon_envelope(q2)
    print("  Q2 carbon pareto:", [(p["label"], round(p["cost"], 1), round(p["carbon"], 2))
                                  for p in cpar["pareto"]], flush=True)
    print("  Q2 envelope:", [(e["epsilon"], e["cost"] and round(e["cost"], 1), e["argmin"])
                             for e in cpar["envelope"]], flush=True)

    print("[T07] Q3 storage (attachment load)", flush=True)
    q3 = run_q3(inputs, cfg)
    print("  Q3 sysd_cost", q3["system"]["d_cost"], "branches", q3["branches"], flush=True)

    print("[T07b] Q3 additional metrics", flush=True)
    from src.q3 import q3_additional_metrics
    q3add = q3_additional_metrics(inputs)
    print("  Q3 add S:", {k: (round(v, 2) if isinstance(v, float) else v)
                          for k, v in q3add["with_storage"].items()}, flush=True)

    print("[T08] Q4 four-cell + interaction", flush=True)
    q4 = run_q4(inputs, cfg, n_schedule=n_sched, t_schedule=t_sched)
    print("  Q4 I=", q4["interaction"]["I"], "F11_src=", q4["F11"]["incumbent_source"], flush=True)

    print("[T08b] carbon epsilon-scan (storage, fixed-task load)", flush=True)
    from src.energy_model import carbon_pareto, make_facility_load
    from src.q4 import _load_to_ai
    aiN = _load_to_ai(inputs, n_sched)
    loadN = make_facility_load(inputs, aiN)
    eps = [0.0, 1.0, 5.0, 10.0, 50.0, 100.0, 300.0, 1000.0, None]
    carp = carbon_pareto(inputs, loadN, eps, with_storage=True)
    Path("results/runs/base/q4/carbon_pareto.json").write_text(
        json.dumps(carp, ensure_ascii=False, indent=1, default=str))
    print("  carbon pareto:", [(p["epsilon"], p["status"],
           (round(p["cost"], 1) if isinstance(p["cost"], (int, float)) else p["cost"]),
           (round(p["carbon"], 3) if isinstance(p["carbon"], (int, float)) else p["carbon"]))
          for p in carp], flush=True)

    print("[T08c] joint storage-aware LNS", flush=True)
    from src.joint_lsn import run_joint_lsn
    lsn = run_joint_lsn(inputs, cfg, n_schedule=n_sched, t_schedule=t_sched, top_k=200)
    print("  LNS best_cost=", lsn["best_cost"], "incumbent=", lsn["best_incumbent"], flush=True)

    print("[T09] scenarios", flush=True)
    sc = run_scenarios(inputs, cfg, n_schedule=n_sched, t_schedule=t_sched)
    print("  transitions", sc["transition_intervals"], flush=True)

    summary = {"elapsed_s": time.time() - t_start, "q1": q1["status"] if isinstance(q1, dict) else q1,
               "q3_system": q3.get("system"), "q4_interaction": q4["interaction"],
               "q2_budgets": {str(k): v["status"] for k, v in (q2["T"] if isinstance(q2, dict) else {}).items()},
               "q2_carbon_envelope": cpar,
               "q3_additional": q3add,
               "scenarios": sc["transition_intervals"]}
    Path("results/summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    print("[done] results/summary.json", flush=True)


if __name__ == "__main__":
    main()