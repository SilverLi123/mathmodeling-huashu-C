# -*- coding: utf-8 -*-
"""任务指派与候选生成（Q1/Q2/Q4 公共）。

容量口径：GPU-hour 平均占用（半开区间），Σ_i g_i·overlap_i(s,d,h) ≤ Available_GPU_r。
任务不可抢占/拆分；实时任务到达即开工；弹性任务整数小时开工 s∈[a_i, 2406−ceil(d_i)]。
"""
import numpy as np
import pandas as pd

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
RIDX = {r: i for i, r in enumerate(REGIONS)}


def _ov(s, d, h):
    lo = max(float(h), float(s))
    hi = min(float(h) + 1.0, float(s) + float(d))
    return max(0.0, hi - lo)


def feasible_regions(inputs, source, maxlat):
    """按时延升序的满足 MaxLatency 目的区域列表 [(region, ms)]。"""
    lat = inputs["latency_long"]
    sub = lat[lat["FromRegion"] == source]
    sub = sub[sub["NetworkLatency_ms"] <= maxlat + 1e-9].sort_values("NetworkLatency_ms")
    return list(zip(sub["ToRegion"], sub["NetworkLatency_ms"].astype(float)))


class TaskScheduler:
    def __init__(self, inputs):
        self.inputs = inputs
        gpu = inputs["gpu"].set_index("Region")
        self.avail = gpu["Available_GPU"].to_numpy(float)
        self.it_cap = gpu["Max_IT_Power_MW"].to_numpy(float)  # IT 功率上限 MW
        self.pw = dict(zip(inputs["mapping"]["TaskType"],
                           inputs["mapping"]["GPU_Power_MW_per_EquivalentGPU"].astype(float)))
        self.wl = inputs["workload"].set_index("TaskID")
        # 非 AI IT 负荷 (region × 2407)，IT 负荷 = AI IT 功率 + NonAI
        rh = inputs["region_hour"]
        nonai = np.zeros((len(REGIONS), 2407))
        for ri, r in enumerate(REGIONS):
            sub = rh[rh["Region"] == r].sort_values("Hour").reset_index(drop=True)
            nonai[ri] = sub["NonAI_IT_Load_MW"].to_numpy(float)
        self.nonai = nonai
        self.ridx = RIDX
        self.used = np.zeros((len(REGIONS), 2407))   # GPU·h 占用
        self.it_mw = np.zeros((len(REGIONS), 2407))  # AI IT 功率 MW
        self.placements = {}  # TaskID -> (region, start)

    def reset(self):
        self.used.fill(0.0)
        self.it_mw.fill(0.0)
        self.placements.clear()

    def _task_power(self, task_id):
        k = self.wl.loc[task_id, "TaskType"]
        return k, self.pw[k]

    def _it_ok(self, region, start, g, d, p):
        """给定任务放置后，AI IT 功率 + NonAI ≤ Max_IT_Power_MW 是否成立。
        （本数据中 Max_Facility = PUE × Max_IT，故设施上限同时满足。）"""
        ri = self.ridx[region]
        cap = self.it_cap[ri]
        t0 = int(np.floor(start))
        for h in range(t0, int(np.ceil(start + d)) + 1):
            if h > 2406 or h < 0:
                continue
            ov = _ov(start, d, h)
            if ov > 0 and self.it_mw[ri, h] + g * ov * p + self.nonai[ri, h] > cap + 1e-4:
                return False
        return True

    def fits(self, region, start, g, d, p):
        ri = self.ridx[region]
        av = self.avail[ri]
        t0 = int(np.floor(start))
        for h in range(t0, int(np.ceil(start + d)) + 1):
            if h > 2406:
                break
            if h < 0:
                continue
            ov = _ov(start, d, h)
            if ov > 0 and self.used[ri, h] + g * ov > av + 1e-4:
                return False
        return self._it_ok(region, start, g, d, p)

    def fits_after_move(self, cur_r, cur_s, new_r, new_s, g, d, p):
        """把任务从 (cur_r,cur_s) 改派到 (new_r,new_s) 后是否仍满足 GPU 与 IT 容量
        （同区域移动时扣除自身原占用）。"""
        ri = self.ridx[new_r]
        av = self.avail[ri]
        cap = self.it_cap[ri]
        t0 = int(np.floor(new_s))
        for h in range(t0, int(np.ceil(new_s + d)) + 1):
            if h > 2406 or h < 0:
                continue
            ovn = _ov(new_s, d, h)
            if ovn <= 0:
                continue
            use_base = self.used[ri, h]
            it_base = self.it_mw[ri, h]
            if cur_r == new_r:
                ovc = _ov(cur_s, d, h)
                use_base -= g * ovc
                it_base -= g * ovc * p
            if use_base + g * ovn > av + 1e-4:
                return False
            if it_base + g * ovn * p + self.nonai[ri, h] > cap + 1e-4:
                return False
        return True

    def add(self, task_id, region, start, g, d):
        ri = self.ridx[region]
        k, p = self._task_power(task_id)
        t0 = int(np.floor(start))
        for h in range(t0, int(np.ceil(start + d)) + 1):
            if h > 2406 or h < 0:
                continue
            ov = _ov(start, d, h)
            if ov > 0:
                self.used[ri, h] += g * ov
                self.it_mw[ri, h] += g * ov * p
        self.placements[task_id] = (region, start)

    def remove(self, task_id, region, start, g, d):
        ri = self.ridx[region]
        _, p = self._task_power(task_id)
        t0 = int(np.floor(start))
        for h in range(t0, int(np.ceil(start + d)) + 1):
            if h > 2406 or h < 0:
                continue
            ov = _ov(start, d, h)
            if ov > 0:
                self.used[ri, h] -= g * ov
                self.it_mw[ri, h] -= g * ov * p
        self.placements.pop(task_id, None)

    def earliest_feasible(self, region, start_lo, g, d, p):
        """从 start_lo 起找最早整数可行开工；无则 None。"""
        latest = 2406 - np.ceil(d) if d <= 2406 else -1
        if latest < start_lo:
            return None
        for s in range(int(np.floor(start_lo)), int(latest) + 1):
            if self.fits(region, s, g, d, p):
                return s
        return None

    def construct(self, wl, order=None, region_policy="least_latency", progress=False):
        """贪婪构造全量可行调度。返回 schedule DataFrame（含 NetworkLatency_ms）。"""
        inputs = self.inputs
        lat = inputs["latency_long"].set_index(["FromRegion", "ToRegion"])["NetworkLatency_ms"]
        self.reset()
        if order is None:
            # 实时优先按到达，弹性按（到达，工作量降序）
            wl = wl.copy()
            wl["_key"] = wl["TaskType"].map({"RealTimeInference": 0, "BatchInference": 1, "AITraining": 2})
            wl["_work"] = wl["GPU_Demand"] * wl["Duration_h"]
            wl = wl.sort_values(["_key", "ArrivalHour", "_work"], ascending=[True, True, False])
        rows = []
        n = len(wl)
        for i, row in enumerate(wl.itertuples(index=False)):
            tid = row.TaskID
            g = float(row.GPU_Demand)
            d = float(row.Duration_h)
            a = float(row.ArrivalHour)
            maxlat = float(row.MaxLatency_ms)
            regs = feasible_regions(inputs, row.SourceRegion, maxlat)
            if row.TaskType == "RealTimeInference":
                # 到达即开工
                placed = False
                for r, ms in regs:
                    if self.fits(r, a, g, d, self.pw[row.TaskType]):
                        self.add(tid, r, a, g, d)
                        rows.append({"TaskID": tid, "AssignedRegion": r, "StartHour": a,
                                     "FinishHour": a + d, "NetworkLatency_ms": ms})
                        placed = True
                        break
                if not placed:
                    return None, (tid, row.TaskType, "realtime_no_feasible_region_at_arrival", a)
            else:
                placed = False
                # 起点：a 起，找最早可行；区域按时延升序
                for r, ms in regs:
                    s = self.earliest_feasible(r, a, g, d, self.pw[row.TaskType])
                    if s is not None:
                        self.add(tid, r, s, g, d)
                        rows.append({"TaskID": tid, "AssignedRegion": r, "StartHour": float(s),
                                     "FinishHour": float(s) + d, "NetworkLatency_ms": ms})
                        placed = True
                        break
                if not placed:
                    return None, (tid, row.TaskType, "flexible_no_feasible_slot", a)
            if progress and i % 5000 == 0:
                print(f"  ...{i}/{n}")
        sch = pd.DataFrame(rows)
        sch = sch.merge(wl[["TaskID", "SourceRegion", "ArrivalHour", "GPU_Demand", "TaskType", "Duration_h"]],
                        on="TaskID", how="left")
        return sch, None

    def build_hourly_gpu(self):
        df = pd.DataFrame(self.used.T, columns=REGIONS)
        df.index.name = "Hour"
        return df

    def build_ai_mw(self, schedule):
        """由 schedule 构造 region×hour 的 AI IT 平均功率 MW。"""
        pw = dict(zip(self.inputs["mapping"]["TaskType"],
                      self.inputs["mapping"]["GPU_Power_MW_per_EquivalentGPU"].astype(float)))
        ai = np.zeros((len(REGIONS), 2407))
        for _, r in schedule.iterrows():
            ri = self.ridx[r["AssignedRegion"]]
            g = float(r["GPU_Demand"]); d = float(r["Duration_h"]); s = float(r["StartHour"])
            p = pw[r["TaskType"]]
            t0 = int(np.floor(s))
            for h in range(t0, int(np.ceil(s + d)) + 1):
                if h > 2406 or h < 0:
                    continue
                ov = _ov(s, d, h)
                if ov > 0:
                    ai[ri, h] += g * ov * p
        df = pd.DataFrame(ai.T, columns=REGIONS)
        df.index.name = "Hour"
        return df

    def build_ai_mw_sched(self, inputs):
        """由 self.placements 直接重建 region×hour AI MW（无需 schedule DataFrame）。"""
        wl = inputs["workload"].set_index("TaskID")
        pw = dict(zip(inputs["mapping"]["TaskType"],
                      inputs["mapping"]["GPU_Power_MW_per_EquivalentGPU"].astype(float)))
        ai = np.zeros((len(REGIONS), 2407))
        for tid, (r, s) in self.placements.items():
            t = wl.loc[tid]
            ri = self.ridx[r]
            g = float(t["GPU_Demand"]); d = float(t["Duration_h"])
            p = pw[t["TaskType"]]
            t0 = int(np.floor(s))
            for h in range(t0, int(np.ceil(s + d)) + 1):
                if h > 2406 or h < 0:
                    continue
                ov = _ov(s, d, h)
                if ov > 0:
                    ai[ri, h] += g * ov * p
        df = pd.DataFrame(ai.T, columns=REGIONS)
        df.index.name = "Hour"
        return df


def candidate_slots(inputs, task, n_starts=8, n_regions=None, granularity="h"):
    """生成某任务的候选 (region, start) 列表，用于邻域 MILP。

    task: Series(TaskType,SourceRegion,ArrivalHour,Duration_h,GPU_Demand,MaxLatency_ms,AssignedRegion,StartHour?).
    realtime: 仅 start=ArrivalHour。flexible: 整数 start ∈ [a, 2406−ceil(d)]，采样 + 最新。
    """
    maxlat = float(task["MaxLatency_ms"])
    regs = feasible_regions(inputs, task["SourceRegion"], maxlat)
    if n_regions is not None:
        regs = regs[:n_regions]
    d = float(task["Duration_h"])
    a = float(task["ArrivalHour"])
    latest = 2406 - np.ceil(d)
    slots = []
    if task["TaskType"] == "RealTimeInference":
        starts = [a]
    else:
        starts = set()
        for s in range(int(np.floor(a)), int(latest) + 1, max(1, n_starts)):
            starts.add(s)
        # 保证包含 a 与 latest 与 a+1..a+min(5,)
        for s in range(int(np.floor(a)), min(int(a) + 8, int(latest) + 1)):
            starts.add(s)
        starts.add(int(latest))
        starts = sorted(s for s in starts if a - 1e-9 <= s <= latest + 1e-9)
        if len(starts) > n_starts + 8:
            # 均匀抽 n_starts + 保留边界
            keep = {starts[0], starts[-1]}
            step = max(1, (len(starts) - 1) // n_starts)
            keep |= {starts[i] for i in range(0, len(starts), step)}
            starts = sorted(keep)
    for r, ms in regs:
        for s in starts:
            slots.append((r, float(s), float(ms)))
    return slots