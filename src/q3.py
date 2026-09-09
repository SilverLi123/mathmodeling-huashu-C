# -*- coding: utf-8 -*-
"""Q3：固定附件负荷下的储能协同优化（LP-first / MILP-guaranteed）。"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.io_data import REGIONS
from src.energy_model import (q3_fixed_load, settle_no_storage, eval_flows,
                              solve_region_storage_lp_first, _region_params)
from src.metrics import storage_metrics


def _region_summary(inputs, flows_sub, with_storage):
    rh = inputs["region_hour"]
    f = flows_sub
    cost = eval_flows(inputs, f)["cost"]
    carbon = eval_flows(inputs, f)["carbon"]
    net = (f["GridPurchase_MW"] - f["GridSell_MW"])
    peak = float(net.max())
    # 按时间顺序（flows 已按 Hour 升序）差分，而非按数值排序差分
    ramp = float(net.diff().abs().mean()) if len(f) > 1 else 0.0
    purchase = float(f["GridPurchase_MW"].sum())
    sold = float(f["GridSell_MW"].sum())
    return {"cost": cost, "carbon": carbon, "peak_net_MW": peak, "mean_abs_ramp_MW": ramp,
            "purchase_MWh": purchase, "sold_MWh": sold}


def run_q3(inputs, cfg, out_dir="results/runs/base/q3"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    load = q3_fixed_load(inputs)
    # N^A：无储能
    flowsN = settle_no_storage(inputs, load)
    enN = eval_flows(inputs, flowsN) if flowsN is not None else {"cost": np.nan, "carbon": np.nan, "ren_util": np.nan}
    # S^A：逐区域 LP-first
    per = {}
    all_flows = []
    for r in REGIONS:
        res = solve_region_storage_lp_first(inputs, r, load[r].to_numpy(float))
        per[r] = {"branch": res.get("branch"), "status": res.get("status"),
                  "cost": res.get("cost"), "carbon": res.get("carbon"),
                  "mutex_viol_n": res.get("mutex_viol_n", None)}
        if res.get("status") in ("optimal",) and res.get("flows") is not None:
            all_flows.append(res["flows"])
    flowsS = pd.concat(all_flows, ignore_index=True) if all_flows else None
    enS = eval_flows(inputs, flowsS) if flowsS is not None else {"cost": np.nan, "carbon": np.nan, "ren_util": np.nan}
    sm = storage_metrics(flowsS, inputs) if flowsS is not None else {}
    # 分区 N vs S 摘要
    per_region = {}
    for r in REGIONS:
        per_region[r] = {
            "N": _region_summary(inputs, flowsN[flowsN["Region"] == r], False),
            "S": _region_summary(inputs, flowsS[flowsS["Region"] == r], True),
        }
    result = {"system": {
        "N_cost": enN.get("cost"), "N_carbon": enN.get("carbon"), "N_util": enN.get("ren_util"),
        "S_cost": enS.get("cost"), "S_carbon": enS.get("carbon"), "S_util": enS.get("ren_util"),
        "d_cost": enS.get("cost", np.nan) - enN.get("cost", np.nan),
        "d_carbon": enS.get("carbon", np.nan) - enN.get("carbon", np.nan),
        "storage_saving": enN.get("cost", np.nan) - enS.get("cost", np.nan),
    }, "per_region": per_region, "branches": {r: per[r]["branch"] for r in REGIONS},
        "storage": sm}
    (out / "metrics_q3.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    if flowsN is not None:
        flowsN.to_csv(out / "hourly_energy_q3_N.csv", index=False)
    if flowsS is not None:
        flowsS.to_csv(out / "hourly_energy_q3_S.csv", index=False)
    return result

def q3_additional_metrics(inputs, out_dir="results/runs/base/q3"):
    """Q3 补充系统级指标（N^A vs S^A）：峰值净购电、平均绝对爬坡、
    新能源利用率、储能吞吐量、储能效率损耗。读 run_q3 已写的逐时 csv。"""
    out = Path(out_dir)
    fN = pd.read_csv(out / "hourly_energy_q3_N.csv")
    fS = pd.read_csv(out / "hourly_energy_q3_S.csv")
    ren_avail = float(inputs["region_hour"]["AvailableRenewable_MW"].sum())
    storage = inputs["storage"].set_index("Region")

    def stat(f):
        p = f.groupby("Hour")["GridPurchase_MW"].sum()
        s_ = f.groupby("Hour")["GridSell_MW"].sum()
        net = (p - s_)
        gross_peak = float(p.max())
        net_peak = float(net.max())
        ramp = float(net.diff().abs().mean())
        ren_used = float((f["RenewableDirect_MW"] + f["RenewableCharge_MW"] + f["RenewableSell_MW"]).sum())
        ren_util = ren_used / ren_avail if ren_avail else float("nan")
        thr = float((f["Charge_MW"] + f["Discharge_MW"]).sum())
        return {"peak_net_purchase_MW": net_peak, "peak_gross_purchase_MW": gross_peak,
                "mean_abs_ramp_MW": ramp, "ren_util": ren_util, "storage_throughput_MWh": thr}

    mN = stat(fN)
    mS = stat(fS)
    loss = 0.0
    if mS["storage_throughput_MWh"] > 0:
        by = fS.groupby("Region").agg(c=("Charge_MW", "sum"), d=("Discharge_MW", "sum"))
        for r in REGIONS:
            eta_c = float(storage.loc[r, "ChargeEfficiency"])
            eta_d = float(storage.loc[r, "DischargeEfficiency"])
            loss += by.loc[r, "c"] * (1 - eta_c) + by.loc[r, "d"] * (1 / eta_d - 1)
    mS["efficiency_loss_MWh"] = float(loss)
    mN["efficiency_loss_MWh"] = 0.0
    payload = {"no_storage": mN, "with_storage": mS,
               "renewable_available_MWh": ren_avail}
    (out / "metrics_q3_additional.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    return payload
