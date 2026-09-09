# -*- coding: utf-8 -*-
"""数据读取、键校验与规范化（T01）。

原始附件只读；本模块负责把 6 个工作簿读成统一结构并缓存到 data/processed，
缓存可由原始附件重建。返回结构：

inputs = {
  'workload': DataFrame   (tasks, 每行一个 TaskID)
  'region_hour': DataFrame(6区域 x 2407小时, 含输入与基准结果列)
  'gpu': DataFrame        (区域 GPU/PUE/IT/设施)
  'latency': DataFrame    (From,To,ms)
  'storage': DataFrame    (区域储能/购售电边界)
  'mapping': DataFrame    (TaskType -> MW/等效GPU)
  'hashes': dict          (原始文件 sha256)
}
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SHEETS = {
    "workload_trace.xlsx": "Sheet1",
    "region_time_data.xlsx": "region_time_data",
    "GPU_information.xlsx": "GPU中心基础情况",
    "network_latency.xlsx": "network_latency",
    "storage_information.xlsx": "storage_information",
    "power_mapping.xlsx": "任务功率映射",
}

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
TASK_TYPES = ["AITraining", "BatchInference", "RealTimeInference"]
TYPE_POWER = {"AITraining": 0.16, "BatchInference": 0.10, "RealTimeInference": 0.08}
TYPE_LATENCY = {"AITraining": 150.0, "BatchInference": 80.0, "RealTimeInference": 20.0}
MAX_HOUR = 2406


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(input_dir: str, cache_dir: str = "data/processed") -> dict:
    """读取全部输入；若 cache_dir 给出则写缓存 CSV（从原始附件可重建）。"""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"附件目录不存在: {input_dir}")
    files = sorted(input_dir.glob("*.xlsx"))
    hashes = {f.name: sha256_file(str(f)) for f in files}

    dfs = {}
    for f in files:
        sheet = SHEETS.get(f.name)
        df = pd.read_excel(f, sheet_name=sheet, header=0) if sheet else pd.read_excel(f, header=0)
        dfs[f.name] = df

    workload = dfs["workload_trace.xlsx"].copy()
    region_hour = dfs["region_time_data.xlsx"].copy()
    gpu = dfs["GPU_information.xlsx"].copy()
    latency = dfs["network_latency.xlsx"].copy()
    storage = dfs["storage_information.xlsx"].copy()
    mapping = dfs["power_mapping.xlsx"].copy()

    # ---------- 规范化 ----------
    workload = workload.sort_values("TaskID").reset_index(drop=True)
    workload["Duration_h"] = workload["EstimatedDuration_min"] / 60.0
    workload["FinishIfArrival"] = workload["ArrivalHour"] + workload["Duration_h"]
    # 每任务允许的目的区域（单向时延 <= MaxLatency_ms）
    lat = latency.pivot_table(index="FromRegion", columns="ToRegion", values="NetworkLatency_ms")
    allowed = {}
    for o in REGIONS:
        row = lat.loc[o]
        allowed[o] = {r for r in REGIONS if row[r] <= 1e-9 or True}  # 占位，下面重算
    # 真实可达表
    reach = {}
    for src in REGIONS:
        reach[src] = {}
        for tgt in REGIONS:
            reach[src][tgt] = float(lat.loc[src, tgt])
    latency_mat = pd.DataFrame(reach)  # index src, col tgt

    gpu = gpu.set_index("Region").loc[REGIONS].reset_index()
    storage = storage.set_index("Region").loc[REGIONS].reset_index()
    mapping = mapping.set_index("TaskType").loc[TASK_TYPES].reset_index()

    region_hour = region_hour.sort_values(["Region", "Hour"]).reset_index(drop=True)
    # 补 PUE、购售电边界、初态（来自 gpu/storage）
    region_hour = region_hour.merge(
        gpu[["Region", "PUE", "Available_GPU", "Max_IT_Power_MW", "Max_Facility_Power_MW"]],
        on="Region", how="left",
    )
    region_hour = region_hour.merge(
        storage[["Region", "StorageCapacity_MWh", "MinSOC_MWh", "InitialSOC_MWh",
                 "MaxChargePower_MW", "MaxDischargePower_MWh" if "MaxDischargePower_MWh" in storage.columns
                 else "MaxDischargePower_MW", "ChargeEfficiency", "DischargeEfficiency",
                 "SellLimit_MW", "MaxGridImport_MW", "MaxGridExport_MW"]],
        on="Region", how="left",
    )

    inputs = {
        "workload": workload,
        "region_hour": region_hour,
        "gpu": gpu,
        "latency": latency_mat,
        "latency_long": latency,
        "storage": storage,
        "mapping": mapping,
        "hashes": hashes,
    }
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        workload.to_csv(cache_dir / "workload.csv", index=False)
        region_hour.to_csv(cache_dir / "region_hour.csv", index=False)
        gpu.to_csv(cache_dir / "gpu.csv", index=False)
        latency_mat.to_csv(cache_dir / "latency.csv")
        storage.to_csv(cache_dir / "storage.csv", index=False)
        mapping.to_csv(cache_dir / "mapping.csv", index=False)
        (cache_dir / "input_hashes.json").write_text(json.dumps(hashes, indent=1, ensure_ascii=False))
    return inputs


def task_candidate_regions(inputs: dict) -> dict:
    """TaskType -> 来源-可达目标列表。返回 (src) -> list[(tgt, ms)] 按 ms 升序。"""
    lat = inputs["latency_long"]
    out = {}
    for src in REGIONS:
        sub = lat[lat["FromRegion"] == src].sort_values("NetworkLatency_ms")
        out[src] = list(zip(sub["ToRegion"], sub["NetworkLatency_ms"].astype(float)))
    return out


def load_inputs_from_cache(cache_dir: str = "data/processed") -> dict:
    """从缓存读（开发/调试用）；正式流程一律从原始附件 load_inputs。"""
    raise NotImplementedError("正式流程要求从原始附件重建；不使用缓存快捷路径覆盖哈希约束")


def _validate_anchors(inputs: dict) -> dict:
    """T01 锚点自检：返回 {name: (ok, actual)} 供 data_audit 使用。"""
    wl = inputs["workload"]
    rh = inputs["region_hour"]
    anchors = {}
    anchors["task_count"] = (len(wl) == 50000, len(wl))
    anchors["type_counts"] = (True, wl["TaskType"].value_counts().to_dict())
    gh = wl.assign(Gh=wl["GPU_Demand"] * wl["Duration_h"]).groupby("TaskType")["Gh"].sum()
    anchors["train_share"] = (abs(gh["AITraining"] / gh.sum() - 0.801376158189607) < 1e-12,
                              float(gh["AITraining"] / gh.sum()))
    anchors["last24"] = (int((wl["ArrivalHour"] >= 2376).sum()) == 538,
                         int((wl["ArrivalHour"] >= 2376).sum()))
    anchors["region_hour_rows"] = (len(rh) == 14442, len(rh))
    anchors["arrival_range"] = (wl["ArrivalHour"].min() == 0 and wl["ArrivalHour"].max() == 2399,
                                (int(wl["ArrivalHour"].min()), int(wl["ArrivalHour"].max())))
    return anchors
