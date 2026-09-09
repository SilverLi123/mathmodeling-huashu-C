# -*- coding: utf-8 -*-
"""Q1 需求预测：18条(区域×类型)小时序列的到达GPU需求预测。

方法：先建均值基准；再以共享特征回归（Ridge），采用**固定起点的直接多步预测**
（direct multi-step with strictly-past features）避免目标/测试期泄漏：

- 训练特征第 t 行只用 series[:t]（不含 y_t 及其后真值）；
- 预测第 st..st+H-1 时，特征统一由 series[:st] 构造（近期均值/滞后取起点处状态），
  仅昼夜谐波随目标时刻变化；不回填任何预测/真实未来值。

指标体系 MAE/RMSE/WAPE（零分母 → NaN，不伪造）。返回 ridge 指标与 mean-baseline 指标。
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
TYPES = ["AITraining", "BatchInference", "RealTimeInference"]

FEATURES = ["mean_hist", "roll_mean_24", "roll_mean_168",
            "lag_1", "lag_24", "lag_168", "sin_h", "cos_h"]
WINS = (24, 168)
LAGS = (1, 24, 168)


def arrival_gpu_series(wl):
    """返回 index=Hour(0..2399), columns=MultiIndex(Region,Type) 的到达GPU需求。"""
    key = wl.groupby(["SourceRegion", "TaskType", "ArrivalHour"])["GPU_Demand"].sum().reset_index()
    idx = pd.MultiIndex.from_product([REGIONS, TYPES], names=["Region", "Type"])
    piv = key.pivot_table(index="ArrivalHour", columns=["SourceRegion", "TaskType"],
                          values="GPU_Demand", aggfunc="sum").reindex(range(2400)).fillna(0.0)
    piv = piv.reindex(columns=idx).fillna(0.0)
    piv.columns = pd.MultiIndex.from_product([REGIONS, TYPES])
    return piv


def _train_features(series):
    """训练特征矩阵：第 t 行只依赖 series[:t]（严格过去，不含 y_t）。"""
    T = len(series)
    cs = np.concatenate([[0.0], np.cumsum(series.astype(float))])  # cs[k]=sum(s[:k])
    cols = {}
    hist = np.zeros(T)
    if T > 1:
        hist[1:] = cs[1:T] / np.arange(1, T)  # hist[t]=cs[t]/t=mean(s[:t])
    cols["mean_hist"] = hist
    for w in WINS:
        rw = np.full(T, np.nan)
        for t in range(w, T):
            rw[t] = (cs[t] - cs[t - w]) / w  # mean(s[t-w:t])，到 t-1 不含 y_t
        cols[f"roll_mean_{w}"] = rw
    for lag in LAGS:
        lg = np.full(T, np.nan)
        if T > lag:
            lg[lag:] = series[:T - lag]
        cols[f"lag_{lag}"] = lg
    h = np.arange(T)
    cols["sin_h"] = np.sin(2 * np.pi * h / 24)
    cols["cos_h"] = np.cos(2 * np.pi * h / 24)
    X = pd.DataFrame(cols, columns=FEATURES)
    return X.ffill().fillna(0.0).to_numpy(float)


def _forecast_features(series, origin, targets):
    """固定起点直接多步：目标时刻 targets 的特征仅由 series[:origin] 构造。"""
    cs = np.concatenate([[0.0], np.cumsum(series.astype(float))])
    T = origin
    base = {"mean_hist": cs[T] / T if T > 0 else 0.0}
    for w in WINS:
        base[f"roll_mean_{w}"] = (cs[T] - cs[T - w]) / w if T >= w else np.nan
    for lag in LAGS:
        base[f"lag_{lag}"] = series[T - lag] if T >= lag else np.nan
    rows = []
    for t in targets:
        row = dict(base)
        row["sin_h"] = np.sin(2 * np.pi * t / 24)
        row["cos_h"] = np.cos(2 * np.pi * t / 24)
        rows.append(row)
    X = pd.DataFrame(rows, columns=FEATURES)
    return X.fillna(0.0).to_numpy(float)


def _wape(y, yhat):
    denom = np.abs(y).sum()
    if denom < 1e-12:
        return np.nan
    return float(np.abs(y - yhat).sum() / denom)


def _fit_predict(Xtr, ytr, Xtt, use_gb):
    if use_gb:
        m = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0)
    else:
        m = Ridge(alpha=1.0)
    m.fit(Xtr, ytr)
    return np.clip(m.predict(Xtt), 0, None)


def backtest_series(series, starts, H, use_gb=False):
    """训练段滚动回测：每个起点 st 用 series[:st] 训练，预测 st..st+H-1（无泄漏）。"""
    records = []
    for st in starts:
        tr = series[:st]
        Xtr = _train_features(tr)
        ytr = tr
        Xtt = _forecast_features(series, st, list(range(st, st + H)))
        ytt = series[st:st + H]
        pred = _fit_predict(Xtr, ytr, Xtt, use_gb)
        meanb = np.full(H, tr.mean())
        records.append({"start": st, "mae_m": float(np.mean(np.abs(ytt - pred))),
                        "rmse_m": float(np.sqrt(np.mean((ytt - pred) ** 2))),
                        "wape_m": _wape(ytt, pred),
                        "mae_base": float(np.mean(np.abs(ytt - meanb))),
                        "rmse_base": float(np.sqrt(np.mean((ytt - meanb) ** 2))),
                        "wape_base": _wape(ytt, meanb)})
    rec = pd.DataFrame(records)
    return {
        "mae_model": float(rec["mae_m"].mean()), "rmse_model": float(rec["rmse_m"].mean()),
        "mae_base": float(rec["mae_base"].mean()), "rmse_base": float(rec["rmse_base"].mean()),
        "wape_model": float(np.nanmean(rec["wape_m"])), "wape_base": float(np.nanmean(rec["wape_base"])),
        "folds": rec,
    }


def run_forecast(wl, cfg):
    """完整预测流程。返回 dict(series, choice, backtest, per_series, predictions,
    metric(ridge最终测试窗), baseline_metric(mean baseline最终测试窗))。"""
    ser = arrival_gpu_series(wl)
    starts = cfg["forecast"]["backtest_starts"]
    H = cfg["forecast"]["horizon"]
    per = {}
    for r in REGIONS:
        for k in TYPES:
            s = ser[(r, k)].to_numpy(float)
            b_ridge = backtest_series(s, starts, H, False)
            per[(r, k)] = {"ridge": b_ridge, "mean_abs": float(np.mean(np.abs(s)))}
    # 回测整体 MAE 判定（Ridge vs 均值基准）
    mae_r = np.nanmean([per[(r, k)]["ridge"]["mae_model"] for r in REGIONS for k in TYPES])
    mae_b = np.nanmean([per[(r, k)]["ridge"]["mae_base"] for r in REGIONS for k in TYPES])
    use_model = mae_r < mae_b
    choice = "ridge" if use_model else "mean_baseline"
    # 最终：0..2375 训练，预测 2376..2399（固定起点多步，无泄漏）
    te_start, te_end = cfg["forecast"]["test_start"], cfg["forecast"]["test_end"]
    preds = []
    metrics = []
    base_metrics = []
    base_abs_err_sum = 0.0
    base_true_sum = 0.0
    for r in REGIONS:
        for k in TYPES:
            s = ser[(r, k)].to_numpy(float)
            tr = s[:te_start]
            yt = s[te_start:te_end + 1]
            pred_base = np.full(H, tr.mean())
            if choice == "ridge":
                pred = _fit_predict(_train_features(tr), tr,
                                    _forecast_features(s, te_start, list(range(te_start, te_end + 1))),
                                    False)
            else:
                pred = np.full(H, tr.mean())
            for j in range(H):
                h = te_start + j
                preds.append({"Region": r, "TaskType": k, "Hour": h,
                              "True_GPU": float(yt[j]), "Pred_GPU": float(pred[j]),
                              "Model": choice, "forecast_from": te_start})
                metrics.append({"Region": r, "TaskType": k, "Hour": h,
                                "MAE": float(np.abs(yt[j] - pred[j])),
                                "SE": float((yt[j] - pred[j]) ** 2)})
                base_metrics.append({"MAE": float(np.abs(yt[j] - pred_base[j])),
                                     "SE": float((yt[j] - pred_base[j]) ** 2)})
                base_abs_err_sum += abs(yt[j] - pred_base[j])
                base_true_sum += abs(yt[j])
    pred_df = pd.DataFrame(preds)
    m_df = pd.DataFrame(metrics)
    b_df = pd.DataFrame(base_metrics)
    wape = float(np.abs(pred_df["True_GPU"] - pred_df["Pred_GPU"]).sum() / pred_df["True_GPU"].abs().sum())
    base_wape = float(base_abs_err_sum / base_true_sum)
    return {"series": ser, "choice": choice, "backtest": per,
            "predictions": pred_df,
            "metric": {"MAE": float(m_df["MAE"].mean()), "RMSE": float(np.sqrt((m_df["SE"].mean()))),
                       "WAPE": wape, "n": len(m_df)},
            "baseline_metric": {"MAE": float(b_df["MAE"].mean()), "RMSE": float(np.sqrt((b_df["SE"].mean()))),
                                "WAPE": base_wape, "n": len(b_df)}}