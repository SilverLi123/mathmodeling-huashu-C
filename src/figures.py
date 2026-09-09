# -*- coding: utf-8 -*-
"""三类候选图生成：raw_/process_/result_ 各 4 幅（覆盖 q1..q4），合计 12 幅。
每图统一导出 SVG + 300DPI PNG。数据全部来自 results/ 与原始附件（不硬编码结论）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "utils"))
SKILL = "/Users/silver/.dsh/.agent-presets/math-modeling/skills/math-modeling"
sys.path.insert(0, str(Path(SKILL) / "references/roles/编程手/scripts"))
import plot_style as ps  # noqa: E402

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
FIGDIR = PROJ / "figures"


def _save(fig, stem, width="report"):
    ps.apply_publication_style("zh", width)
    stem = FIGDIR / stem
    fig.savefig(str(FIGDIR / (Path(stem).name + ".svg")))
    fig.savefig(str(FIGDIR / (Path(stem).name + ".png")), dpi=300)
    plt.close(fig)


def _inputs():
    from src.io_data import load_inputs
    return load_inputs("C题 面向算电协同的多目标调度优化研究/附件数据", cache_dir=None)


# ---------------------------------------------------------------------------
# raw 类
# ---------------------------------------------------------------------------
def raw_q1_arrival_profile():
    from src.forecast import arrival_gpu_series
    inp = _inputs()
    ser = arrival_gpu_series(inp["workload"])
    fig, axes = ps.publication_subplots(2, 3, width="double", aspect=0.9)
    from src.forecast import TYPES
    cols = {"AITraining": ps.PALETTE["primary"], "BatchInference": ps.PALETTE["secondary"],
            "RealTimeInference": ps.PALETTE["contrast"]}
    for i, r in enumerate(REGIONS):
        ax = axes[i // 3][i % 3]
        for k in TYPES:
            s = ser[(r, k)].to_numpy(float)
            ax.plot(range(2400), s, lw=0.6, color=cols[k], label=k.replace("Inference", ""))
        ax.set_title(r.replace("Region", "区"))
        if i // 3 == 1:
            ax.set_xlabel("小时")
        if i % 3 == 0:
            ax.set_ylabel("到达GPU需求")
        ax.set_ylim(bottom=0)
        if i == 0:
            ax.legend(frameon=False, fontsize=6)
    fig.suptitle("区域×类型 小时到达GPU需求（0–2399）", x=0.01, ha="left", fontsize=9)
    _save(fig, "raw_q1_arrival_profile", "double")


def raw_q2_energy_params():
    inp = _inputs()
    rh = inp["region_hour"]
    reg = "RegionD"
    sub = rh[rh["Region"] == reg].sort_values("Hour")
    fig, axes = ps.publication_subplots(3, 1, width="report", aspect=1.1, height_ratios=[1, 1, 1])
    h = sub["Hour"].to_numpy()
    axes[0].plot(h, sub["ElectricityPrice_CNY_per_MWh"], lw=0.8, color=ps.PALETTE["primary"])
    axes[0].set_ylabel("购电价 元/MWh")
    axes[0].set_title("RegionD 购电价")
    axes[1].plot(h, sub["CarbonIntensity_tCO2_per_MWh"], lw=0.8, color=ps.PALETTE["secondary"])
    axes[1].set_ylabel("碳强度 t/MWh")
    axes[1].set_title("碳强度")
    axes[2].plot(h, sub["AvailableRenewable_MW"], lw=0.8, color=ps.PALETTE["positive"])
    axes[2].axhline(float(inp["storage"].set_index("Region").loc[reg, "SellLimit_MW"]), lw=0.8,
                    color=ps.PALETTE["contrast"], label="外送上限")
    axes[2].set_ylabel("可用新能源 MW")
    axes[2].set_xlabel("小时")
    axes[2].legend(frameon=False, fontsize=7)
    axes[2].set_title("可用新能源（六区逐时相同）")
    _save(fig, "raw_q2_energy_params")


def raw_q3_load_vs_renewable():
    from src.energy_model import q3_fixed_load
    inp = _inputs()
    load = q3_fixed_load(inp)
    ren = inp["region_hour"].pivot_table(index="Hour", columns="Region", values="AvailableRenewable_MW")
    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1])
    for i, r in enumerate(REGIONS):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        m = (ren[r] - load[r]).to_numpy()
        ax.plot(np.arange(2407), m, lw=0.6, color=ps.PALETTE["primary"])
        ax.axhline(0, color=ps.PALETTE["neutral"], lw=0.6)
        ax.fill_between(np.arange(2407), m, 0, where=m < 0, color=ps.PALETTE["contrast"], alpha=0.4)
        ax.set_title(f"{r.replace('Region','区')}  裕度∈[{m.min():.0f},{m.max():.0f}]MW", fontsize=8)
        if i // 3 == 1:
            ax.set_xlabel("小时")
        if i % 3 == 0:
            ax.set_ylabel("R−L (MW)")
    axz = fig.add_subplot(gs[2, :])
    hz = np.arange(2390, 2407)
    axz.plot(hz, ren["RegionF"].to_numpy()[2390:2407], lw=1.2, color=ps.PALETTE["positive"], label="可用新能源")
    axz.plot(hz, load["RegionF"].to_numpy()[2390:2407], lw=1.2, color=ps.PALETTE["primary"], label="固定负荷")
    axz.scatter([2400], [load["RegionF"].iloc[2400]], s=28, color=ps.PALETTE["contrast"], zorder=5, label="F@2400 缺口")
    axz.set_xlabel("小时"); axz.set_ylabel("MW"); axz.legend(frameon=False, fontsize=7)
    axz.set_title("RegionF 2390–2406 放大：唯一负裕度出现在 2400")
    fig.suptitle("六区域新能源裕度 R−L 与 RegionF@2400 缺口放大", x=0.01, ha="left", fontsize=10)
    _save(fig, "raw_q3_load_vs_renewable", "double")


def raw_q4_storage_config():
    inp = _inputs()
    st = inp["storage"]
    x = np.arange(6)
    fig, ax = ps.publication_subplots(1, 1, width="report", aspect=0.7)
    ax.bar(x - 0.2, st["StorageCapacity_MWh"], 0.35, color=ps.PALETTE["primary"], label="容量 MWh")
    ax.bar(x + 0.2, st["InitialSOC_MWh"], 0.35, color=ps.PALETTE["positive"], label="初始SOC MWh")
    ax.set_xticks(x); ax.set_xticklabels([r.replace("Region", "") for r in REGIONS])
    ax.set_ylabel("MWh"); ax.legend(frameon=False, fontsize=7)
    ax.set_title("各区域储能容量与初始SOC")
    _save(fig, "raw_q4_storage_config")


# ---------------------------------------------------------------------------
# process 类
# ---------------------------------------------------------------------------
def process_q1_utilization_and_forecast():
    inp = _inputs()
    hg = pd.read_csv(PROJ / "results/runs/base/q1/hourly_gpu_q1.csv", index_col=0)
    avail = inp["gpu"].set_index("Region")["Available_GPU"]
    fig, axes = ps.publication_subplots(2, 1, width="report", aspect=1.2, height_ratios=[1.2, 1])
    for r, c in zip(["RegionD", "RegionA"], [ps.PALETTE["primary"], ps.PALETTE["secondary"]]):
        axes[0].plot(hg.index, hg[r].to_numpy() / avail[r] * 100, lw=0.7, color=c, label=r.replace("Region", "区"))
    axes[0].set_ylabel("GPU利用率 %"); axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[0].set_title("基础调度的逐时GPU利用率（区A、D）")
    # 预测 1 个序列
    pred = pd.read_csv(PROJ / "results/q1_predictions.csv") if (PROJ / "results/q1_predictions.csv").exists() else None
    if pred is not None:
        s = pred[(pred.Region == "RegionD") & (pred.TaskType == "AITraining")]
        axes[1].plot(s["Hour"], s["True_GPU"], marker="o", ms=3, lw=1, color=ps.PALETTE["primary"], label="真实")
        axes[1].plot(s["Hour"], s["Pred_GPU"], marker="s", ms=3, lw=1, color=ps.PALETTE["contrast"], label="预测")
        axes[1].set_ylabel("到达GPU需求"); axes[1].set_xlabel("小时 2376–2399")
        axes[1].legend(frameon=False, fontsize=7)
        axes[1].set_title("RegionD·AITraining 预测 vs 真实（最终测试窗）")
    _save(fig, "process_q1_util_and_forecast")


def process_q2_tradeoff():
    m = json.load(open(PROJ / "results/runs/base/q2/metrics_q2.json"))
    pts = [("N", m["N"]["energy"]["cost"], m["N"]["energy"]["carbon"], m["N"]["sched_metrics"]["mean_latency_ms"])]
    for lb in [20.0, 80.0, 150.0]:
        v = m["T"].get(str(lb)) or m["T"].get(lb)
        if v and v["status"] == "OK":
            pts.append((f"T{int(lb)}", v["energy"]["cost"], v["energy"]["carbon"], v["sched_metrics"]["mean_latency_ms"]))
    fig, axes = ps.publication_subplots(1, 2, width="double", aspect=0.62)
    ax = axes[0]
    for (label, c, car, lat) in pts:
        ax.scatter([lat], [c / 1e6], s=30, color=ps.PALETTE["primary"])
        ax.annotate(label, (lat, c / 1e6), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("平均网络时延 ms"); ax.set_ylabel("总成本 百万元")
    ax.set_title("成本-时延折中")
    ax2 = axes[1]
    for (label, c, car, lat) in pts:
        ax2.scatter([c / 1e6], [car], s=30, color=ps.PALETTE["contrast"])
        ax2.annotate(label, (c / 1e6, car), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax2.set_xlabel("总成本 百万元"); ax2.set_ylabel(r"碳排 tCO$_2$")
    ax2.set_title("成本-碳折中")
    _save(fig, "process_q2_tradeoff", "double")


def process_q3_soc_and_mc():
    from src.energy_model import q3_fixed_load
    inp = _inputs()
    load = q3_fixed_load(inp)
    fS = pd.read_csv(PROJ / "results/runs/base/q3/hourly_energy_q3_S.csv")
    sub = fS[fS.Region == "RegionF"].sort_values("Hour").reset_index(drop=True)
    rhF = inp["region_hour"][inp["region_hour"].Region == "RegionF"].sort_values("Hour").reset_index(drop=True)
    C = sub["Charge_MW"].to_numpy(float); D = sub["Discharge_MW"].to_numpy(float)
    win = 48
    best_s = int(np.argmax(np.convolve(C + D, np.ones(win), "valid")))
    h = np.arange(best_s, best_s + win)
    sl = slice(best_s, best_s + win)
    fig, axes = ps.publication_subplots(4, 1, width="report", aspect=1.05, height_ratios=[1, 1, 1, 1])
    axes[0].plot(h, rhF["ElectricityPrice_CNY_per_MWh"].to_numpy()[sl], lw=1.0, color=ps.PALETTE["primary"])
    axes[0].set_ylabel("购电价"); axes[0].set_title(f"RegionF 48h 代表窗口（{best_s}–{best_s+win}）电价")
    axes[1].plot(h, (rhF["AvailableRenewable_MW"] - load["RegionF"]).to_numpy()[sl], lw=1.0, color=ps.PALETTE["positive"])
    axes[1].axhline(0, color=ps.PALETTE["neutral"], lw=0.6); axes[1].set_ylabel("R−L (MW)")
    axes[1].set_title("新能源裕度")
    axes[2].plot(h, C[sl], lw=1.0, color=ps.PALETTE["positive"], label="充电")
    axes[2].plot(h, D[sl], lw=1.0, color=ps.PALETTE["contrast"], label="放电")
    axes[2].set_ylabel("MW"); axes[2].legend(frameon=False, fontsize=7); axes[2].set_title("充放电功率")
    axes[3].plot(h, sub["SOC_MWh"].to_numpy()[sl], lw=1.0, color=ps.PALETTE["primary"])
    axes[3].set_ylabel("MWh"); axes[3].set_xlabel("小时"); axes[3].set_title("储能 SOC")
    _save(fig, "process_q3_soc_and_mc")


def process_q4_transition():
    sc = pd.read_csv(PROJ / "results/runs/scenarios/scenario_renewable_scale.csv")
    fig, ax = ps.publication_subplots(1, 1, width="report", aspect=0.8)
    rho = sc["rho"]
    ax.plot(rho, sc["task_saving"] / 1e6, marker="o", ms=4, color=ps.PALETTE["primary"], label="任务独立增益 F00−F10（≥0）")
    ax.plot(rho, sc["storage_saving_fixed"] / 1e6, marker="s", ms=4, color=ps.PALETTE["contrast"], label="储能节省 F00−F01（恒正）")
    ax.plot(rho, sc["I"] / 1e6, marker="^", ms=4, color=ps.PALETTE["positive"], label="交互项 I")
    ax.axhline(0, color=ps.PALETTE["neutral"], lw=0.7)
    ax.set_xlabel("新能源规模系数 ρ"); ax.set_ylabel("百万元")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("随 ρ 下降：任务迁移独立增益衰减至 0、储能节省恒正、交互为替代")
    _save(fig, "process_q4_transition")


# ---------------------------------------------------------------------------
# result 类
# ---------------------------------------------------------------------------
def result_q1_gantt():
    g = pd.read_csv(PROJ / "results/runs/base/q1/schedule_last24.csv")
    typecol = {"RealTimeInference": ps.PALETTE["contrast"], "BatchInference": ps.PALETTE["secondary"],
               "AITraining": ps.PALETTE["primary"]}
    fig, axes = ps.publication_subplots(2, 3, width="double", aspect=1.0)
    for i, r in enumerate(REGIONS):
        ax = axes[i // 3][i % 3]
        sub = g[g.AssignedRegion == r]
        for j, (_, row) in enumerate(sub.iterrows()):
            ax.barh(0, row["FinishHour"] - row["StartHour"], left=row["StartHour"], height=1,
                    color=typecol[row["TaskType"]], edgecolor="none", alpha=0.8)
        ax.set_xlim(2375.8, 2406.2)
        ax.set_yticks([])
        ax.set_title(r.replace("Region", "区") + f"  (n={len(sub)})")
        ax.set_xlabel("小时" if i // 3 == 1 else "")
    fig.suptitle("最后24小时到达任务调度甘特图（0–2406 收尾）", x=0.01, ha="left", fontsize=9)
    _save(fig, "result_q1_gantt", "double")


def result_q2_migration_and_effect():
    schT = pd.read_csv(PROJ / "results/runs/base/q2/schedule_q2_lb150.csv")
    mm = schT.groupby(["SourceRegion", "AssignedRegion"]).size().unstack(fill_value=0)
    mm = mm.reindex(index=REGIONS, columns=REGIONS, fill_value=0)
    M = mm.to_numpy().astype(float)
    M = np.where(np.eye(6, dtype=bool), np.nan, M)   # 去对角
    rs = np.nansum(M, axis=1, keepdims=True)
    M = np.where(rs > 0, M / rs * 100, np.nan)        # 行归一（占该来源迁移任务比例 %）
    fig, axes = ps.publication_subplots(1, 2, width="double", aspect=0.6, width_ratios=[1, 1.2])
    ax = axes[0]
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(6)); ax.set_xticklabels([r.replace("Region", "") for r in REGIONS], fontsize=6)
    ax.set_yticks(range(6)); ax.set_yticklabels([r.replace("Region", "") for r in REGIONS], fontsize=6)
    ax.set_xlabel("目标区域"); ax.set_ylabel("来源区域")
    ax.set_title("迁移任务去向比例（去对角、行归一，150ms）")
    fig.colorbar(im, ax=ax, fraction=0.046)
    # O/N/T 购电与碳
    inp = _inputs()
    rh = inp["region_hour"]
    O_pur = rh["GridPurchase_MW"].sum()
    fN = pd.read_csv(PROJ / "results/runs/base/q1/hourly_energy_q1.csv")
    fT = pd.read_csv(PROJ / "results/runs/base/q2/hourly_energy_q2_lb150.csv")
    N_pur = fN["GridPurchase_MW"].sum(); T_pur = fT["GridPurchase_MW"].sum()
    ax2 = axes[1]
    x = np.arange(3)
    ax2.bar(x - 0.2, [O_pur / 1e6, N_pur / 1e6, T_pur / 1e6], 0.4, color=ps.PALETTE["primary"])
    ax2.set_xticks(x); ax2.set_xticklabels(["O原始", "N固定", "T任务"])
    ax2.set_ylabel("总购电 百万MWh")
    ax2.set_title("购电量：原始基准 vs 统一口径")
    _save(fig, "result_q2_migration_and_effect", "double")


def result_q3_storage_value():
    m = json.load(open(PROJ / "results/runs/base/q3/metrics_q3.json"))
    regs = REGIONS
    vals = [m["per_region"][r]["N"]["cost"] - m["per_region"][r]["S"]["cost"] for r in regs]
    fig, ax = ps.publication_subplots(1, 1, width="report", aspect=0.72)
    colors = [ps.PALETTE["contrast"] if v > 0 else ps.PALETTE["neutral"] for v in vals]
    ax.bar([r.replace("Region", "") for r in regs], np.array(vals) / 1e6, color=colors)
    ax.axhline(0, color=ps.PALETTE["neutral"], lw=0.7)
    ax.set_ylabel("纯储能节省 F(N)−F(S) 百万元")
    ax.set_title("分区域储能价值（ABC无外送→无价值；DEF外送时序套利）")
    _save(fig, "result_q3_storage_value")


def result_q4_fourcell():
    m = json.load(open(PROJ / "results/runs/base/q4/metrics_q4.json"))
    F00 = m["F00"]["energy"]["cost"] / 1e6
    F10 = m["F10"]["energy"]["cost"] / 1e6
    F01 = m["F01"]["energy"]["cost"] / 1e6
    F11 = m["F11"]["energy"]["cost"] / 1e6
    B_T = F00 - F10   # 任务迁移单独收益
    B_S = F00 - F01   # 储能单独收益
    B_J = F00 - F11   # 联合实际收益
    I = m["interaction"]["I"] / 1e6
    cats = ["任务迁移\n单独收益", "储能\n单独收益", "独立叠加\n(假设)", "联合\n实际收益", "交互项\n(替代)"]
    vals = [B_T, B_S, B_T + B_S, B_J, I]
    cols = [ps.PALETTE["positive"], ps.PALETTE["primary"], ps.PALETTE["secondary"],
            ps.PALETTE["contrast"], ps.PALETTE["neutral"]]
    fig, ax = ps.publication_subplots(1, 1, width="report", aspect=0.78)
    bars = ax.bar(cats, vals, color=cols)
    bars[2].set_alpha(0.45)
    ax.axhline(0, color=ps.PALETTE["neutral"], lw=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (1.2 if v >= 0 else -2.6),
                f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_ylabel("收益 百万元")
    ax.set_title(f"四格收益分解：联合=储能（任务迁移收益被吸收），交互 I={I:.1f} 百万元")
    _save(fig, "result_q4_fourcell")


def result_q1_utilization():
    inp = _inputs()
    hg = pd.read_csv(PROJ / "results/runs/base/q1/hourly_gpu_q1.csv", index_col=0)
    avail = inp["gpu"].set_index("Region")["Available_GPU"]
    fig, axes = ps.publication_subplots(2, 3, width="double", aspect=0.95)
    for i, r in enumerate(REGIONS):
        ax = axes[i // 3][i % 3]
        ax.plot(hg.index, hg[r].to_numpy() / avail[r] * 100, lw=0.6, color=ps.PALETTE["primary"])
        ax.set_title(r.replace("Region", "区") + f" 峰值{100*hg[r].max()/avail[r]:.1f}%")
        if i // 3 == 1:
            ax.set_xlabel("小时")
        if i % 3 == 0:
            ax.set_ylabel("GPU利用率 %")
        ax.set_ylim(0, 100)
    fig.suptitle("基础调度的分区域逐时 GPU 利用率", x=0.01, ha="left", fontsize=9)
    _save(fig, "result_q1_utilization", "double")


def result_q2_cost_carbon():
    inp = _inputs()
    rh = inp["region_hour"]
    O_c = float((rh["ElectricityPrice_CNY_per_MWh"] * rh["GridPurchase_MW"]).sum()
                - (rh["SellPrice_CNY_per_MWh"] * rh["GridSell_MW"]).sum())
    O_e = float(rh["CarbonEmission_tCO2"].sum())
    m1 = json.load(open(PROJ / "results/runs/base/q1/metrics_q1.json"))
    m2 = json.load(open(PROJ / "results/runs/base/q2/metrics_q2.json"))
    N_c, N_e = m1["energy"]["cost"], m1["energy"]["carbon"]
    t = m2["T"].get("150") or m2["T"].get("150.0")
    T_c, T_e = t["energy"]["cost"], t["energy"]["carbon"]
    fig, axes = ps.publication_subplots(1, 2, width="double", aspect=0.6)
    x = np.arange(3)
    axes[0].bar(x, [O_c / 1e6, N_c / 1e6, T_c / 1e6], color=[ps.PALETTE["neutral"], ps.PALETTE["primary"], ps.PALETTE["contrast"]])
    axes[0].set_xticks(x); axes[0].set_xticklabels(["O 原始", "N 固定", "T 任务"])
    axes[0].set_ylabel("总成本 百万元"); axes[0].set_title("成本对比")
    axes[1].bar(x, [O_e / 1e6, N_e, T_e], color=[ps.PALETTE["neutral"], ps.PALETTE["primary"], ps.PALETTE["contrast"]])
    axes[1].set_xticks(x); axes[1].set_xticklabels(["O 原始", "N 固定", "T 任务"])
    axes[1].set_ylabel(r"碳排 tCO$_2$"); axes[1].set_title("碳排对比")
    _save(fig, "result_q2_cost_carbon", "double")


def result_q3_system():
    m = json.load(open(PROJ / "results/runs/base/q3/metrics_q3.json"))
    fig, ax = ps.publication_subplots(1, 1, width="report", aspect=0.7)
    lab = ["N(固定)", "S(储能)"]
    vals = [m["system"]["N_cost"] / 1e6, m["system"]["S_cost"] / 1e6]
    ax.bar(lab, vals, color=[ps.PALETTE["primary"], ps.PALETTE["positive"]])
    ax.set_ylabel("系统成本 百万元")
    ax.set_title(f"储能使成本 -{m['system']['N_cost']/1e6:.1f}→-{m['system']['S_cost']/1e6:.1f} 百万元（碳 {m['system']['N_carbon']:.1f}→{m['system']['S_carbon']:.1f} tCO$_2$）")
    _save(fig, "result_q3_system")


def result_q4_scenario():
    sc = json.load(open(PROJ / "results/runs/scenarios/scenarios.json"))
    fig, axes = ps.publication_subplots(1, 2, width="double", aspect=0.62)
    for ax, key, ttl in [(axes[0], "low_renewable_outage", "短时新能源缺口"),
                         (axes[1], "price_spread", "外送价×0.7、购电价×1.3")]:
        d = sc["special"][key]
        cells = ["F00", "F10", "F01", "F11"]
        ax.bar(cells, [d[k] / 1e6 for k in cells],
               color=[ps.PALETTE["primary"], ps.PALETTE["secondary"], ps.PALETTE["positive"], ps.PALETTE["contrast"]])
        ax.set_ylabel("总成本 百万元"); ax.set_title(ttl)
    _save(fig, "result_q4_scenario", "double")


def process_q2_carbon():
    d = json.load(open(PROJ / "results/runs/base/q2/carbon_pareto/carbon_envelope_q2.json"))
    pts = d["points"]
    par = sorted(d["pareto"], key=lambda p: p["carbon"])
    fig, ax = ps.publication_subplots(1, 1, width="report", aspect=0.8)
    for p in pts:
        ax.scatter(p["carbon"], p["cost"] / 1e6, s=48, color=ps.PALETTE["primary"], zorder=3)
        ax.annotate(p["label"], (p["carbon"], p["cost"] / 1e6),
                    textcoords="offset points", xytext=(4, -14), fontsize=8)
    xs = [p["carbon"] for p in par]
    ys = [p["cost"] / 1e6 for p in par]
    ax.step(xs, ys, where="post", color=ps.PALETTE["contrast"], lw=1.8, label="ε-约束下包络")
    ax.set_xlabel(r"碳排 tCO$_2$")
    ax.set_ylabel("总成本 百万元（负值）")
    ax.set_title("无储能 Cost-Carbon 前沿：碳预算越松成本越低（阶梯）")
    ax.legend(loc="best", fontsize=8, frameon=False)
    _save(fig, "process_q2_carbon")


def make_all():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for fn in [raw_q1_arrival_profile, raw_q2_energy_params, raw_q3_load_vs_renewable, raw_q4_storage_config,
               process_q1_utilization_and_forecast, process_q2_tradeoff, process_q2_carbon, process_q3_soc_and_mc,
               process_q4_transition, result_q1_gantt, result_q2_migration_and_effect,
               result_q3_storage_value, result_q4_fourcell,
               result_q1_utilization, result_q2_cost_carbon, result_q3_system, result_q4_scenario]:
        try:
            fn()
            print("OK ", fn.__name__, flush=True)
        except Exception as e:
            print("ERR", fn.__name__, repr(e), flush=True)


if __name__ == "__main__":
    make_all()