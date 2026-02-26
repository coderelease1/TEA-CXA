#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import jsonlines
import time
import hashlib
import threading
import requests
from typing import List, Dict, Any, Iterable
from datetime import datetime
import datasets

# ========== 配置区域 ==========
SERVERS_HOST_FILE = "../servers_node.txt"
DEFAULT_SERVERS_HOST = os.getenv("SERVERS_HOST", "127.0.0.1")

# 复用你已有的轮询思想：多个 report_generation 服务端口
REPORT_PORTS = ["5008"]  # 可扩展，例如 ["5008", "5011", "5012"]

# 缓存文件（JSONL，append-only）
CACHE_JSONL_PATH = "./medversa_cache_chexbench.jsonl"

# 超时设置
HTTP_TIMEOUT = 80

# =================================

_rr_index = 0
_rr_lock = threading.Lock()

def read_server_host_from_file(path: str) -> str:
    if not os.path.exists(path):
        return DEFAULT_SERVERS_HOST
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            return s
    return DEFAULT_SERVERS_HOST

SERVERS_HOST = read_server_host_from_file(SERVERS_HOST_FILE)

def get_next_report_port() -> str:
    global _rr_index
    with _rr_lock:
        port = REPORT_PORTS[_rr_index % len(REPORT_PORTS)]
        _rr_index += 1
        return port

def normalize_image_paths(image_paths: List[str]) -> List[str]:
    # 排序后返回，用绝对路径或你数据集中稳定的相对路径
    return sorted([str(p) for p in image_paths])

def make_key(image_paths_sorted: List[str]) -> str:
    # 使用 SHA256 对排序后的列表进行哈希，得到稳定 key
    h = hashlib.sha256()
    for p in image_paths_sorted:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()

def load_cache_index(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    index = {}
    if not os.path.exists(jsonl_path):
        return index
    with jsonlines.open(jsonl_path, "r") as reader:
        for obj in reader:
            k = obj.get("key")
            if k:
                index[k] = obj
    return index

def append_cache_record(jsonl_path: str, record: Dict[str, Any]) -> None:
    # 追加写入一行
    with jsonlines.open(jsonl_path, mode="a") as writer:
        writer.write(record)

def call_report_service(image_paths_sorted: List[str], context: str) -> Any:
    payload = {
        "image_paths": image_paths_sorted,
        "context": context
    }
    headers = {"Content-Type": "application/json"}
    proxies = {"http": None, "https": None}
    port = get_next_report_port()
    url = f"http://{SERVERS_HOST}:{port}/report_generation"

    resp = requests.post(url, json=payload, headers=headers, proxies=proxies, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def precompute_for_dataset(
    samples: Iterable[Dict[str, Any]],
    cache_jsonl_path: str = CACHE_JSONL_PATH
):
    """
    samples: 可迭代对象，每个样本形如：
        {
          "image_paths": ["/path/a.png", "/path/b.png", ...],  # 一条数据的所有图像
          "context": "Age:52. Gender:M. Indication: ..."
        }
    """
    cache_index = load_cache_index(cache_jsonl_path)

    total = 0
    hit = 0
    miss = 0
    start_time = time.time()

    for i in range(len(samples)):
        sample = samples[i]
        total += 1
        image_paths = sample["images"]
        context = ''#sample.get("x-ray info").get("context")

        img_sorted = image_paths #normalize_image_paths(image_paths)
        key = make_key(img_sorted)

        if key in cache_index:
            hit += 1
            continue

        try:
            result = call_report_service(img_sorted, context)#[os.path.join("/your/path",path) for path in img_sorted]
            record = {
                "key": key,                
                "report": result,
                "image_paths": img_sorted,
                "context": context,
                #"timestamp": datetime.utcnow().isoformat() + "Z"
            }
            append_cache_record(cache_jsonl_path, record)
            cache_index[key] = record
            miss += 1
        except requests.exceptions.Timeout:
            print(f"[Timeout] {img_sorted}")
        except requests.exceptions.ConnectionError:
            print(f"[ConnError] {img_sorted}")
        except requests.exceptions.RequestException as e:
            detail = getattr(e, "response", None)
            detail_text = detail.text if detail is not None else "No detail"
            print(f"[HTTPError] {e} | detail: {detail_text} | {img_sorted}")
        except Exception as e:
            print(f"[Error] {type(e).__name__}: {e} | {img_sorted}")

    dur = time.time() - start_time
    print(f"Done. total={total}, cache_hit={hit}, new_cached={miss}, time_sec={dur:.1f}")

if __name__ == "__main__":
    # 你需要把数据集的“每条数据的所有图像路径 + context”提供给 samples 迭代器
    # 下面仅演示结构，实际请替换为你数据加载逻辑
    """ demo_samples = [
        {
            "image_paths": [
                "/your/path/deid_png_8bit_new/GRDN9MSQUAU4T974/GRDNQGIFW08TWWIL/studies/1.2.826.../series/.../instances/...1.png",
                "/your/path/deid_png_8bit_new/GRDN9MSQUAU4T974/GRDNQGIFW08TWWIL/studies/1.2.826.../series/.../instances/...2.png",
            ],
            "context": "Age:52. Gender:M. Indication: Shortness of breath for 2 days."
        },
        # 更多样本...
    ] """
    dataframes = []
    data_files = ['../chexbench/chexbench_processed.jsonl']
    for parquet_file in data_files:

        dataframe = datasets.load_dataset("json", data_files=parquet_file)["train"]
        dataframes.append(dataframe)
    dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

    print(f"dataset len: {len(dataframe)}")

    precompute_for_dataset(dataframe, CACHE_JSONL_PATH)