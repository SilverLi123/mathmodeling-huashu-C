# -*- coding: utf-8 -*-
"""统一能源结算与储能优化（Q2/Q3/Q4 公共）。稀疏 LP/MILP（HiGHS via scipy）。

口径（与附件1一致）：
  逐时非负功率：新能源直供 Rload、新能源充电 Rch、新能源外送 Rsell、弃电 Rcurt；
               电网供负荷 Gload、电网充电 Gch；储能放电 D；总充电 C=Rch+Gch。
  购电 B=Gload+Gch，售电 P_sell=Rsell（仅新能源可外送；电网购电与放电不可转售）。
  能量守恒：Rload+Rch+Rsell+Rcurt=R；Rload+Gload+D=L。
  电费 F=Σ(π·B−π_sell·P_sell)；碳 E=Σ c·B；利用率 η=Σ(Rload+Rch+Rsell)/ΣR。
  SOC(t)=SOC(t−1)+ηc·C(t)−D(t)/ηd；SOC(2406)≥InitialSOC。

无储能解析（前提：购电上限不绑定且 π_buy≥π_sell）：
  B=(L−R)^+；P_sell=min((R−L)^+,X)；弃电=(R−L−P_sell)^+。
  边际成本 MC：弃电段=0；外送段(0<余量≤X)=π_sell；购电段(L>R)=π_buy。
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
T_HOURS = 2407  # 小时 0..2406
# 带储能每区域每时段变量索引
K_RLOAD, K_RCH, K_RSELL, K_RCURT, K_GL, K_GCH, K_D, K_SOC = range(8)
PER_S = 8
# 无储能变量索引
N_RLOAD, N_RSELL, N_RCURT, N_GL = range(4)
PER_N = 4
MUTEX_TOL = 1e-5


def _region_params(inputs, region):
    g = inputs["gpu"].set_index("Region").loc[region]
    s = inputs["storage"].set_index("Region").loc[region]
    rh = inputs["region_hour"]
    sub = rh[rh["Region"] == region].sort_values("Hour").reset_index(drop=True)
    assert len(sub) == 2407, region
    return {
        "buy": sub["ElectricityPrice_CNY_per_MWh"].to_numpy(float),
        "sell": sub["SellPrice_CNY_per_MWh"].to_numpy(float),
        "carb": sub["CarbonIntensity_tCO2_per_MWh"].to_numpy(float),
        "ren": sub["AvailableRenewable_MW"].to_numpy(float),
        "import_cap": float(s["MaxGridImport_MW"]),
        "export_cap": float(min(s["MaxGridExport_MW"], s["SellLimit_MW"])),
        "cap": float(s["StorageCapacity_MWh"]),
        "soc_min": float(s["MinSOC_MWh"]),
        "soc0": float(s["InitialSOC_MWh"]),
        "cmax": float(s["MaxChargePower_MW"]),
        "dmax": float(s["MaxDischargePower_MW"]),
        "eta_c": float(s["ChargeEfficiency"]),
        "eta_d": float(s["DischargeEfficiency"]),
        "pue": float(g["PUE"]),
    }


def no_storage_settle_hour(L, R, X, buy_p, sell_p, c_int, import_cap):
    """单区域-小时无储能解析结算。feasible=False: 缺口超购电上限。"""
    excess = R - L
    if excess >= 0.0:
        sell = min(excess, X)
        curt = max(excess - sell, 0.0)
        B, mc, note = 0.0, (0.0 if excess > X + 1e-9 else sell_p), "surplus"
    else:
        need = -excess
        if need > import_cap + 1e-6:
            return {"feasible": False, "note": "import_cap_violated", "B": None}
        B, sell, curt, mc, note = need, 0.0, 0.0, buy_p, "deficit"
    cost = B * buy_p - sell * sell_p
    return {"B": B, "Rsell": sell, "Rcurt": curt, "cost": cost,
            "carbon": B * c_int, "mc": mc, "feasible": True, "note": note}


def no_storage_mc(load_arr, ren_arr, sell_p_arr, buy_p_arr, X):
    """向量化无储能边际成本（元/MWh）：弃电=0 / 外送=π_sell / 购电=π_buy。"""
    ex = ren_arr - load_arr
    return np.where(ex > X + 1e-9, 0.0, np.where(ex > 1e-9, sell_p_arr, buy_p_arr))


def settle_no_storage(inputs, load_df, region_subset=None):
    """Q2 口径无储能逐小时解析结算（全时域含2406）。返回长表 flows 或 None。"""
    regions = region_subset or REGIONS
    prms = {r: _region_params(inputs, r) for r in regions}
    rows = []
    for t in range(2407):
        for r in regions:
            p = prms[r]
            s = no_storage_settle_hour(float(load_df.loc[t, r]), p["ren"][t], p["export_cap"],
                                       p["buy"][t], p["sell"][t], p["carb"][t], p["import_cap"])
            if not s["feasible"]:
                return None
            l = float(load_df.loc[t, r])
            rload = min(l, p["ren"][t])
            rows.append({"Region": r, "Hour": t, "Load_MW": l,
                         "RenewableDirect_MW": rload, "RenewableCharge_MW": 0.0,
                         "RenewableSell_MW": s["Rsell"], "Curtailment_MW": s["Rcurt"],
                         "GridToLoad_MW": s["B"], "GridCharge_MW": 0.0,
                         "GridPurchase_MW": s["B"], "GridSell_MW": s["Rsell"],
                         "Charge_MW": 0.0, "Discharge_MW": 0.0, "SOC_MWh": np.nan})
    return pd.DataFrame(rows)


def eval_flows(inputs, flows):
    """由 flows 长表独立复算 cost(元)/carbon(tCO2)/ren_util。"""
    if flows is None or len(flows) == 0:
        return {"cost": np.nan, "carbon": np.nan, "ren_util": np.nan}
    rh = inputs["region_hour"].set_index(["Region", "Hour"])
    f = flows.set_index(["Region", "Hour"])
    bp = rh["ElectricityPrice_CNY_per_MWh"].reindex(f.index).to_numpy(float)
    sp = rh["SellPrice_CNY_per_MWh"].reindex(f.index).to_numpy(float)
    ci = rh["CarbonIntensity_tCO2_per_MWh"].reindex(f.index).to_numpy(float)
    rn = rh["AvailableRenewable_MW"].reindex(f.index).to_numpy(float)
    B = f["GridPurchase_MW"].to_numpy(float)
    S = f["GridSell_MW"].to_numpy(float)
    used = (f["RenewableDirect_MW"] + f["RenewableCharge_MW"] + f["GridSell_MW"]).to_numpy(float)
    return {"cost": float((B * bp - S * sp).sum()), "carbon": float((B * ci).sum()),
            "ren_util": float(used.sum() / rn.sum()) if rn.sum() > 0 else np.nan}


def _flows_from_x(x, regions, load_df, with_storage, n_cols, per, n_reg):
    rows = []
    for ri, r in enumerate(regions):
        for t in range(2407):
            base = ri * 2407 * per + t * per
            if with_storage:
                Rload, Rch, Rsell, Rcurt = x[base + K_RLOAD], x[base + K_RCH], x[base + K_RSELL], x[base + K_RCURT]
                Gload, Gch, D, SOC = x[base + K_GL], x[base + K_GCH], x[base + K_D], x[base + K_SOC]
            else:
                Rload, Rsell, Rcurt = x[base + N_RLOAD], x[base + N_RSELL], x[base + N_RCURT]
                Gload = x[base + N_GL]
                Rch = Gch = D = 0.0
                SOC = np.nan
            rows.append({"Region": r, "Hour": t, "Load_MW": float(load_df.loc[t, r]),
                         "RenewableDirect_MW": Rload, "RenewableCharge_MW": Rch,
                         "RenewableSell_MW": Rsell, "Curtailment_MW": Rcurt,
                         "GridToLoad_MW": Gload, "GridCharge_MW": Gch,
                         "GridPurchase_MW": Gload + Gch, "GridSell_MW": Rsell,
                         "Charge_MW": Rch + Gch, "Discharge_MW": D, "SOC_MWh": SOC})
    return pd.DataFrame(rows)


def solve_energy(inputs, load_df, with_storage=True, carbon_cap=None, region_subset=None,
                 mutex_binary=False, min_throughput=False, cost_tau=None, cost_cap=None,
                 budget_s=120.0, mip_gap=1e-4):
    """统一系统级能源/储能优化。

    返回 dict：status('optimal'/'infeasible'/'timeout')、obj、flows、x、
               is_milp、mutex_viol_n、mip_gap、message。
    mutex_binary=True → 每区域每小时 z 二进制（充放电互斥）。
    min_throughput=True → 目标改为最小化充放电吞吐（ΣC+D）。
    cost_cap=标量 → 追加主成本上界行 Σ(πB−πs·Rsell) ≤ cost_cap。
    """
    regions = region_subset or list(load_df.columns)
    per = PER_S if with_storage else PER_N
    n_reg = len(regions)
    n_z = n_reg * 2407 if (mutex_binary and with_storage) else 0
    n_cols = n_reg * 2407 * per + n_z
    prms = {r: _region_params(inputs, r) for r in regions}

    def col(ri, t, k):
        return ri * 2407 * per + t * per + k

    obj = np.zeros(n_cols)
    Ieq, Jeq, Veq, beq = [], [], [], []
    Iub, Jub, Vub, bub = [], [], [], []

    def add_eq(coefs, rhs):
        r = len(beq)
        for c_, v in coefs.items():
            Ieq.append(r); Jeq.append(c_); Veq.append(v)
        beq.append(rhs)

    def add_ub(coefs, rhs):
        r = len(bub)
        for c_, v in coefs.items():
            Iub.append(r); Jub.append(c_); Vub.append(v)
        bub.append(rhs)

    for ri, r in enumerate(regions):
        p = prms[r]
        L = load_df[r].to_numpy(float)
        for t in range(2407):
            base = col(ri, t, 0)
            zcol = n_cols - n_z + ri * 2407 + t
            if with_storage:
                if min_throughput:
                    obj[base + K_RCH] = 1.0
                    obj[base + K_GCH] = 1.0
                    obj[base + K_D] = 1.0
                else:
                    obj[base + K_RSELL] = -p["sell"][t]
                    obj[base + K_GL] = p["buy"][t]
                    obj[base + K_GCH] += p["buy"][t]
                # 可再生守恒
                add_eq({base + K_RLOAD: 1, base + K_RCH: 1, base + K_RSELL: 1, base + K_RCURT: 1}, p["ren"][t])
                # 负荷平衡
                add_eq({base + K_RLOAD: 1, base + K_GL: 1, base + K_D: 1}, float(L[t]))
                # 购电上限
                add_ub({base + K_GL: 1, base + K_GCH: 1}, p["import_cap"])
                # 外送上限
                add_ub({base + K_RSELL: 1}, p["export_cap"])
                # 充电功率上限（互斥 z：z=1 允许 C≤cmax；z=0 强制 C=0）
                cd = {base + K_RCH: 1, base + K_GCH: 1}
                if mutex_binary:
                    cd[zcol] = -p["cmax"]
                    add_ub(cd, 0.0)
                else:
                    add_ub(cd, p["cmax"])
                # 放电上限
                dd = {base + K_D: 1}
                if mutex_binary:
                    dd[zcol] = p["dmax"]
                add_ub(dd, p["dmax"])
                # SOC 上下限
                add_ub({base + K_SOC: 1}, p["cap"])
                add_ub({base + K_SOC: -1}, -p["soc_min"])
                # SOC 递推
                rec = {base + K_SOC: 1, base + K_RCH: -p["eta_c"], base + K_GCH: -p["eta_c"],
                       base + K_D: 1.0 / p["eta_d"]}
                if t > 0:
                    rec[base - per + K_SOC] = -1
                add_eq(rec, p["soc0"] if t == 0 else 0.0)
            else:
                obj[base + N_RSELL] = -p["sell"][t]
                obj[base + N_GL] = p["buy"][t]
                add_eq({base + N_RLOAD: 1, base + N_RSELL: 1, base + N_RCURT: 1}, p["ren"][t])
                add_eq({base + N_RLOAD: 1, base + N_GL: 1}, float(L[t]))
                add_ub({base + N_GL: 1}, p["import_cap"])
                add_ub({base + N_RSELL: 1}, p["export_cap"])
    if with_storage:
        for ri, r in enumerate(regions):
            p = prms[r]
            add_ub({col(ri, 2406, K_SOC): -1}, -p["soc0"])
    if carbon_cap is not None:
        cd = {}
        for ri, r in enumerate(regions):
            p = prms[r]
            for t in range(2407):
                base = col(ri, t, 0)
                if with_storage:
                    cd[base + K_GL] = p["carb"][t]
                    cd[base + K_GCH] = p["carb"][t]
                else:
                    cd[base + N_GL] = p["carb"][t]
        add_ub(cd, carbon_cap)
    if cost_cap is not None:
        cd = {}
        for ri, r in enumerate(regions):
            p = prms[r]
            for t in range(2407):
                base = col(ri, t, 0)
                if with_storage:
                    cd[base + K_GL] = p["buy"][t]
                    cd[base + K_GCH] = p["buy"][t]
                    cd[base + K_RSELL] = -p["sell"][t]
                else:
                    cd[base + N_GL] = p["buy"][t]
                    cd[base + N_RSELL] = -p["sell"][t]
        add_ub(cd, cost_cap)
    # 装配
    n_eq = len(beq)
    A_eq = coo_matrix((Veq, (Ieq, Jeq)), shape=(n_eq, n_cols)).tocsr()
    A_ub = coo_matrix((Vub, (Iub, Jub)), shape=(len(bub), n_cols)).tocsr()
    lo = np.zeros(n_cols)
    hi = np.full(n_cols, np.inf)
    if mutex_binary and with_storage:
        hi[n_cols - n_z:] = 1.0  # 模式变量 z ∈ {0,1}
        integrality = np.zeros(n_cols)
        integrality[n_cols - n_z:] = 1
        res = milp(c=obj, constraints=[LinearConstraint(A_ub, -np.inf, np.array(bub)),
                                       LinearConstraint(A_eq, np.array(beq), np.array(beq))],
                   integrality=integrality, bounds=Bounds(lo, hi),
                   options={"time_limit": budget_s, "mip_rel_gap": mip_gap})
        status = {0: "optimal", 1: "iteration_limit", 2: "infeasible", 3: "unbounded", 4: "other"}.get(res.status, "other")
        if res.status != 0:
            return {"status": status, "message": res.message, "obj": None}
        return _finish(res.x, regions, load_df, with_storage, n_cols, per, prms, is_milp=True,
                       mip_gap=getattr(res, "mip_gap", None), status=status)
    res = linprog(c=obj, A_ub=A_ub, b_ub=np.array(bub), A_eq=A_eq, b_eq=np.array(beq),
                  bounds=list(zip(lo, hi)), method="highs", options={"time_limit": budget_s})
    status = {0: "optimal", 1: "iteration_limit", 2: "infeasible", 3: "unbounded", 4: "other"}.get(res.status, "other")
    if res.status != 0:
        return {"status": status, "message": res.message, "obj": None}
    return _finish(res.x, regions, load_df, with_storage, n_cols, per, prms, is_milp=False,
                   status=status)


def _finish(x, regions, load_df, with_storage, n_cols, per, prms, is_milp, status, mip_gap=None):
    flows = _flows_from_x(x, regions, load_df, with_storage, n_cols, per, len(regions))
    cost = 0.0
    carbon = 0.0
    used = 0.0
    total_ren = 0.0
    for _, row in flows.iterrows():
        p = prms[row["Region"]]
        t = int(row["Hour"])
        cost += row["GridPurchase_MW"] * p["buy"][t] - row["GridSell_MW"] * p["sell"][t]
        carbon += row["GridPurchase_MW"] * p["carb"][t]
        used += row["RenewableDirect_MW"] + row["RenewableCharge_MW"] + row["GridSell_MW"]
        total_ren += p["ren"][t]
    mutex_viol = 0
    if with_storage and len(flows):
        f = flows
        C = f["Charge_MW"].to_numpy(float)
        D = f["Discharge_MW"].to_numpy(float)
        mutex_viol = int((np.minimum(C, D) > MUTEX_TOL).sum())
    return {"status": status, "obj": float(cost), "cost": float(cost), "carbon": float(carbon),
            "ren_util": float(used / total_ren) if total_ren > 0 else np.nan,
            "flows": flows, "x": x, "is_milp": is_milp, "mip_gap": mip_gap,
            "mutex_viol_n": mutex_viol, "message": ""}


def solve_region_storage_lp_first(inputs, region, load, budget_s=120.0):
    """Q3 单区域 LP-first 流程：
    1) LP（无互斥 z）→ 检查 min(C,D)≤1e-5 MW 全部通过 → branch='lp_first'；
    2) 存在循环 → 主成本上界 cost≤LP*+τ 面内次级最小吞吐 → 再检查 → branch='lp_first+secondary'；
    3) 仍违反 → MILP（z 二进制）→ branch='milp'。
    返回 dict(status, branch, cost, carbon, flows, mutex_viol_n, secondary_used, tau, message)。
    """
    load_df = pd.DataFrame({region: load})
    r1 = solve_energy(inputs, load_df, with_storage=True, mutex_binary=False,
                      budget_s=budget_s, region_subset=[region])
    if r1["status"] != "optimal":
        return r1
    # 判定是否为空载循环：仅当既无互斥违例、又几乎无充放吞吐时才视为干净（lp_first）
    thr = 0.0
    if r1["flows"] is not None:
        thr = float((r1["flows"]["Charge_MW"].to_numpy(float)
                     + r1["flows"]["Discharge_MW"].to_numpy(float)).sum())
    if r1["mutex_viol_n"] == 0 and thr <= 1e-2:
        r1["branch"] = "lp_first"
        r1["secondary_used"] = False
        return r1
    # 次级：吞吐目标 + 近精确成本上界（消除零价值/互斥空循环，保留经济有效搬运）
    tau = max(1e-3, 1e-6 * abs(r1["cost"]))
    r2 = solve_energy(inputs, load_df, with_storage=True, mutex_binary=False,
                      min_throughput=True, cost_cap=r1["cost"] + tau,
                      budget_s=budget_s, region_subset=[region])
    if r2["status"] == "optimal" and r2["mutex_viol_n"] == 0:
        r2["branch"] = "lp_first+secondary"
        r2["secondary_used"] = True
        r2["tau"] = tau
        r2["cost_first"] = r1["cost"]          # LP 主目标下界
        r2["cost_secondary"] = r2["cost"]      # 次级解实际主目标成本（与 flows 一致）
        return r2
    # MILP 回退
    r3 = solve_energy(inputs, load_df, with_storage=True, mutex_binary=True,
                      budget_s=budget_s, region_subset=[region])
    r3["branch"] = "milp"
    r3["secondary_used"] = False
    r3["cost_first"] = r1["cost"]
    return r3


def solve_region_storage_milp(inputs, region, load, budget_s=120.0, cost_ref=None):
    """单区域储能 MILP（z 二进制保证互斥）。"""
    load_df = pd.DataFrame({region: load})
    r = solve_energy(inputs, load_df, with_storage=True, mutex_binary=True,
                     budget_s=budget_s, region_subset=[region])
    if r["status"] == "optimal":
        r["branch"] = "milp"
        r["secondary_used"] = False
    return r


def q3_fixed_load(inputs):
    """附件固定设施负荷 DataFrame（region×hour）。"""
    rh = inputs["region_hour"]
    cols = {}
    for r in REGIONS:
        sub = rh[rh["Region"] == r].sort_values("Hour").reset_index(drop=True)
        L = (sub["Baseline_AI_IT_Load_MW"] + sub["NonAI_IT_Load_MW"]) * sub["PUE"]
        cols[r] = L.to_numpy(float)
    return pd.DataFrame(cols, index=np.arange(2407))


def make_facility_load(inputs, ai_mw_df):
    """由 AI MW(region×hour) 构造设施负荷：PUE×(NonAI+AI)。"""
    rh = inputs["region_hour"]
    cols = {}
    for r in REGIONS:
        sub = rh[rh["Region"] == r].sort_values("Hour").reset_index(drop=True)
        nonai = sub["NonAI_IT_Load_MW"].to_numpy(float)
        pue = float(sub["PUE"].iloc[0])
        ai = ai_mw_df[r].to_numpy(float) if r in ai_mw_df.columns else np.zeros(2407)
        cols[r] = (nonai + ai) * pue
    return pd.DataFrame(cols, index=np.arange(2407))


def carbon_pareto(inputs, load_df, epsilons, with_storage=True, region_subset=None,
                  budget_s=120.0):
    """ε-constraint 碳预算扫描：对同一负荷体系逐 ε 解 min F s.t. E ≤ ε，
    产出 (cost, carbon) 帕累托样本（ε=None 表示无碳上限=成本极小点）。
    用于 Q2/Q4 的成本-碳权衡实证，替代"退化"式先验断言。"""
    pts = []
    for eps in epsilons:
        r = solve_energy(inputs, load_df, with_storage=with_storage,
                         carbon_cap=eps, region_subset=region_subset, budget_s=budget_s)
        pts.append({"epsilon": eps, "status": r["status"],
                    "cost": r["cost"], "carbon": r["carbon"]})
    return pts
