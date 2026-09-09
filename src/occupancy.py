# -*- coding: utf-8 -*-
"""任务占用与功率映射（T03 公共）。

半开区间约定：任务在时刻 s 开工、持续 d 小时，占用小时 t 的重叠为
    overlap_hours(s, d, t) = max(0, min(t+1, s+d) - max(t, s))      （单位 h）
本模块不依赖求解器内部结构；导出 schedule.csv 后由 checks.py 独立复算。
"""
import numpy as np
import pandas as pd

HOURS_ELEC = np.arange(0, 2407, dtype=float)  # 电力结算小时 0..2406
ARRIVAL_TMAX = 2406.0  # 任何任务不得占用 [2406,2407)


def overlap_hours(start: float, duration: float, hour: int) -> float:
    s, d, t = float(start), float(duration), int(hour)
    lo, hi = max(t, s), min(t + 1.0, s + d)
    return max(0.0, hi - lo)


def overlap_vec(starts, durations, hours):
    """向量化 overlap_hours：starts/durations 数组 × hours 数组 -> (n,h) 矩阵。"""
    starts = np.asarray(starts, dtype=float)
    durations = np.asarray(durations, dtype=float)
    hours = np.asarray(hours, dtype=float)
    lo = np.maximum.outer(starts, hours)  # (n,h)
    hi = np.minimum.outer(starts + durations, hours + 1.0)
    return np.maximum(0.0, hi - lo)


class OccupancyBuilder:
    """逐小时 GPU 平均占用/功率累加器。

    用法：b = OccupancyBuilder(region_hour, gpu, mapping)
          b.add_tasks(task_df, schedule_df)   # schedule: TaskID,AssignedRegion,StartHour
          b.hourly_gpu(), b.hourly_ai_mw()
    """

    def __init__(self, inputs: dict):
        self.inputs = inputs
        self.regions = list(inputs["gpu"]["Region"])
        self.reg_index = {r: i for i, r in enumerate(self.regions)}
        n_r, n_h = len(self.regions), 2407
        self._gpu = np.zeros((n_r, n_h))
        self._ai = np.zeros((n_r, n_h))
        self.type_power = dict(zip(inputs["mapping"]["TaskType"],
                                   inputs["mapping"]["GPU_Power_MW_per_EquivalentGPU"].astype(float)))

    def _add(self, gpu_req, power, region, start, duration, hour_min, hour_max):
        pass  # 具体见 add_tasks 的向量化实现

    def add_tasks(self, task_df: pd.DataFrame, sched_df: pd.DataFrame) -> None:
        """task_df 需含列 TaskID,GPU_Demand,TaskType,Duration_h；
        sched_df 含列 TaskID,AssignedRegion,StartHour。"""
        m = task_df.merge(sched_df[["TaskID", "AssignedRegion", "StartHour"]], on="TaskID", how="inner")
        if len(m) != len(task_df):
            missing = set(task_df["TaskID"]) - set(m["TaskID"])
            raise ValueError(f"schedule 缺失任务 {len(missing)} 个，如 {sorted(missing)[:5]}")
        m = m.sort_values("TaskID")
        g = m["GPU_Demand"].to_numpy(dtype=float)
        pw = m["TaskType"].map(self.type_power).to_numpy(dtype=float)
        reg = m["AssignedRegion"].map(self.reg_index).to_numpy(dtype=int)
        s = m["StartHour"].to_numpy(dtype=float)
        d = m["Duration_h"].to_numpy(dtype=float)
        # 构造 (task, hour) 稀疏重叠
        t0 = np.floor(s).astype(int)
        # 每任务最多占用 ceil(d)+2 个小时
        max_sp = int(np.ceil(np.max(d))) + 2
        for k in range(max_sp):
            hour = t0 + k
            if np.all(hour > 2406):
                break
            valid = hour <= 2406
            if not np.any(valid):
                break
            idx = np.nonzero(valid)[0]
            ov = overlap_hours_vec(s[idx], d[idx], hour[idx])
            np.add.at(self._gpu, (reg[idx], hour[idx]), g[idx] * ov)
            np.add.at(self._ai, (reg[idx], hour[idx]), g[idx] * pw[idx] * ov)
        # 2406 小时不允许 AI 占用（收尾检查由 checks 完成）

    def hourly_gpu(self) -> pd.DataFrame:
        rr = self.inputs["gpu"]["Region"].tolist()
        return pd.DataFrame(self._gpu.T, columns=rr)  # index hour

    def hourly_ai_mw(self) -> pd.DataFrame:
        rr = self.inputs["gpu"]["Region"].tolist()
        return pd.DataFrame(self._ai.T, columns=rr)


def overlap_hours_vec(starts, durations, hours):
    """向量化按元素：len(starts)==len(hours)。"""
    s = np.asarray(starts, dtype=float)
    d = np.asarray(durations, dtype=float)
    t = np.asarray(hours, dtype=float)
    lo = np.maximum(t, s)
    hi = np.minimum(t + 1.0, s + d)
    return np.maximum(0.0, hi - lo)


def gpu_workload_by_region_type(wl: pd.DataFrame) -> pd.DataFrame:
    """主时域(0-2399)区域×类型 到达GPU需求 / GPU·h / 任务数。"""
    rows = []
    for (r, k), sub in wl.groupby(["SourceRegion", "TaskType"]):
        rows.append({
            "SourceRegion": r, "TaskType": k, "TaskCount": len(sub),
            "ArrivalGPU": float(sub["GPU_Demand"].sum()),
            "ArrivalGPUh": float((sub["GPU_Demand"] * sub["Duration_h"]).sum()),
        })
    return pd.DataFrame(rows).sort_values(["SourceRegion", "TaskType"])


def instant_occupancy_if_arrival(wl: pd.DataFrame) -> pd.DataFrame:
    """按来源到达即执行（无迁移）得到的逐时 GPU 平均占用（主时域诊断用）。"""
    rows = []
    for (r, k), sub in wl.groupby(["SourceRegion", "TaskType"]):
        t0 = np.floor(sub["ArrivalHour"].to_numpy(dtype=float)).astype(int)
        for j in range(int(np.ceil(sub["Duration_h"].max())) + 2):
            h = t0 + j
            valid = h <= 2399
            if not np.any(valid):
                break
            idx = np.nonzero(valid)[0]
            ov = overlap_hours_vec(sub["ArrivalHour"].to_numpy(dtype=float)[idx],
                                   sub["Duration_h"].to_numpy(dtype=float)[idx], h[idx])
            rows.append(pd.DataFrame({
                "SourceRegion": r, "TaskType": k, "Hour": h[idx],
                "GPU_Occupancy": sub["GPU_Demand"].to_numpy(dtype=float)[idx] * ov,
            }))
    df = pd.concat(rows, ignore_index=True)
    piv = df.pivot_table(index="Hour", columns=["SourceRegion", "TaskType"],
                         values="GPU_Occupancy", aggfunc="sum").fillna(0.0)
    return piv.reindex(range(2400)).fillna(0.0)
