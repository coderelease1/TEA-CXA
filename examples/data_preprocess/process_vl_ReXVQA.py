# -*- coding: utf-8 -*-
# Preprocess Chest X-ray VQA dataset to JSONL with context and view info
# Randomly subsample train=2000, test=500 by default.

import argparse
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union, Set

import datasets

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("preprocess_chest_vqa")

" "
instruction_following = (
    r"You are a medical imaging expert. Given a question and chest X-ray images, output the best answer choice (A/B/C/D). "
    r"Patient context is provided within <context></context>."
    "\nYou have access to multiple tools; each may make mistakes. Prefer calling at least one tool per question. Use diverse tools as needed, cross-validate results, and integrate them with your own reasoning. Do not call the same tool with identical parameters more than once.\n"
    r"Any time you receive new information, you should reason step by step inside the <think> and </think> tags. "
    r"Then you can either call one or more tool functions or provide the final answer choice. "
    "Return the final choice as a single letter wrapped in <answer></answer>. Example: <answer>A</answer>."
)

# The real image root prefix provided by user
IMAGE_ROOT_PREFIX = "/aifs4su/xmed/zahuai/datasets/deid_png_8bit_new"

def save_dataset_as_jsonl_no_escaped_slash(dataset: datasets.Dataset, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in dataset:
            line = json.dumps(ex, ensure_ascii=False)
            line = line.replace("\\/", "/")
            f.write(line + "\n")

def read_any_json_dataset(path: str) -> List[Dict[str, Any]]:
    """ # Auto-detect json or jsonl
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jsonl", ".json"]:
        ds = datasets.load_dataset("json", data_files=path)["train"]
        return ds
    else:
        raise ValueError(f"Unsupported file extension for {path}. Use .json or .jsonl") """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:        
        obj = json.load(f)
        for k, v in obj.items():
            if isinstance(v, dict):

                v = {"id": k, **v}
                records.append(v)

    return records

def norm_age_to_bucket(patient_age: Optional[str]) -> Tuple[str, str]:
    """
    Convert PatientAge like '058Y' or '58Y' or '58' to:
    - age_years_str: '58' (or 'Unknown')
    - age_bucket_str: '50-60' (or 'Unknown')
    Rules:
      - If endswith Y/y, strip and parse.
      - If pure digits, parse directly.
      - Otherwise Unknown.
    """
    if not patient_age or not isinstance(patient_age, str):
        return "Unknown", "Unknown"
    s = patient_age.strip()
    if s.endswith(("Y", "y")):
        s = s[:-1]
    # strip leading zeros
    s = s.lstrip("0")
    if s == "":
        s = "0"
    if s.isdigit():
        try:
            age = int(s)
            low = (age // 10) * 10
            high = low + 10
            bucket = f"{low}-{high}"
            return str(age), bucket
        except Exception:
            return "Unknown", "Unknown"
    return "Unknown", "Unknown"

def build_context(example: Dict[str, Any]) -> str:
    """
    Build context string like:
      "<context>Age:50-60.Gender:M.Indication: ___-year-old male with history of metastatic melanoma, presenting with confusion and somnolence.  Evaluate for acute cardiopulmonary process.</context>"
    Fallback to Unknown if missing.
    """
    age_years, age_bucket = norm_age_to_bucket(example.get("PatientAge"))
    sex = str(example.get("PatientSex") or "Unknown").strip() or "Unknown"
    indication = example.get("Indication")
    if indication is None:
        indication = "Unknown"
    else:
        indication = str(indication).strip() or "Unknown"
    context_lines = [
        f"Age:"+ age_years + ".",
        f"Gender:{sex}.",
        f"Indication: {indication}."
    ]
    return " ".join(context_lines)

def join_question_with_options(question: str, options: List[str], image_paths, context: Optional[str]) -> str:
    """
    Compose final question text as:
      {context}\nQuestion:{question}\nA) ...\nB) ...\n...
    Options in source could be like "A. text", "B) text", etc. We will normalize to "A) text".
    """
    q = '<image>' * len(image_paths) + question.strip()
    # normalize options
    norm_opts = []
    for opt in options:
        s = str(opt).strip()
        # try to split leading label
        # cases: "A. foo", "A) foo", "A - foo", or without label
        label = None
        if len(s) >= 2 and s[1] in [".", ")", ":", "-"]:
            candidate = s[0].upper()
            if candidate in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                label = candidate
                s = s[2:].strip()
        # fallback: try first token 'A.' or 'A)'
        if label is None and len(s) >= 3 and s[1].isspace() and s[0].upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            label = s[0].upper()
            s = s[2:].strip()
        if label is None:
            # try detect "A " then text
            parts = s.split(maxsplit=1)
            if parts and len(parts[0]) == 1 and parts[0].upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                label = parts[0].upper()
                s = parts[1] if len(parts) > 1 else ""
        if label is None:
            # cannot detect, append without forced label (rare)
            norm_opts.append(s)
        else:
            norm_opts.append(f"{label}) {s}")
    context_block = (context.strip() + "\n") if context else ""
    full = f"{context_block}Question:{q}\n" + "\n".join(norm_opts)
    return full

def fix_image_paths(image_paths: List[str]) -> List[str]:
    """
    Convert relative entries like '../deid_png/GRDN.../...png' to absolute real paths:
    /aifs4su/.../deid_png/GRDN.../...png
    Strategy: find the subpath starting from 'deid_png' and join with IMAGE_ROOT_PREFIX.
    """
    fixed = []
    for p in image_paths:
        p = str(p)
        # normalize separators
        p = p.replace("\\", "/")
        if "deid_png/" in p:
            sub = p.split("deid_png/")[-1]
            fixed.append(os.path.join(IMAGE_ROOT_PREFIX, sub))
        else:
            # If already absolute and starts with IMAGE_ROOT_PREFIX, keep
            if p.startswith(IMAGE_ROOT_PREFIX):
                fixed.append(p)
            else:
                # fallback: join anyway
                fixed.append(os.path.join(IMAGE_ROOT_PREFIX, p.lstrip("/")))
    return fixed

def pick_answer_letter(example: Dict[str, Any]) -> Optional[str]:
    # The key may be "correct_answer" or "answer" depending on source
    for k in ["correct_answer", "answer", "CorrectAnswer", "Correct_Answer"]:
        if k in example and example[k]:
            v = str(example[k]).strip()
            # keep only first char if like "C" or "C. ..."
            if len(v) > 1 and v[1] in [".", ")", ":"]:
                return v[0].upper()
            return v[0].upper()
    return None

def get_options(example: Dict[str, Any]) -> List[str]:
    opts = example.get("options")
    if isinstance(opts, list) and len(opts) > 0:
        return [str(x) for x in opts]
    # if options absent, try to reconstruct from other fields (not typical)
    return []

def get_question_text(example: Dict[str, Any]) -> str:
    q = example.get("question")
    return str(q) if q is not None else ""

def get_image_list(example: Dict[str, Any]) -> List[str]:
    paths = example.get("ImagePath") or example.get("images") or []
    if isinstance(paths, list):
        return [str(x) for x in paths]
    elif isinstance(paths, str):
        return [paths]
    else:
        return []

def get_view_position(example: Dict[str, Any]) -> Union[str, List[str]]:
    vp = example.get("ImageViewPosition")
    assert isinstance(vp, list)

    return vp

def map_example(example: Dict[str, Any], split: str) -> Optional[Dict[str, Any]]:
    try:
        context = build_context(example)
        question = get_question_text(example)
        options = get_options(example)
        image_paths = fix_image_paths(get_image_list(example))
        img_view = get_view_position(example)

        prompt_text = join_question_with_options(question, options, image_paths, "<context>" + context + "</context>")
        ans_letter = pick_answer_letter(example)
     

        data = {
            "id": example.get("id"),
            "prompt": [
                {
                    "role": "system",
                    "content": instruction_following
                },
                {
                    "role": "user",
                    "content": f"{prompt_text}"
                },
            ],
            "images": image_paths,
            "reward_model": {"style": "rule", "ground_truth": ans_letter},
            "x-ray info": {
                "context": context,
                "ImageViewPosition": img_view
            },
            "extra_info": {
                "split": split
            },
        }
        return data
    except Exception as e:
        logger.warning(f"Skip example due to error: {e}")
        return None

""" def subsample_dataset(ds: datasets.Dataset, n: int, seed: int) -> datasets.Dataset:
    n = min(n, len(ds))
    if n == len(ds):
        return ds.shuffle(seed=seed)  # keep random order but full size
    idx = list(range(len(ds)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    pick = idx[:n]
    return ds.select(pick) """

def subsample_dataset(records: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:  # CHANGED
    n = min(n, len(records))
    idx = list(range(len(records)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    pick = idx[:n]
    # 保持随机顺序
    return [records[i] for i in pick]

def map_records(records: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out = []
    for ex in records:
        mapped = map_example(ex, split)
        if mapped is not None:
            out.append(mapped)
    return out

def read_id_list(txt_path: Optional[str]) -> Set[str]:  # CHANGED/NEW
    """
    从txt文件读取每行一个id，返回去重后的集合。
    如果路径为空或文件不存在，返回空集合。
    """
    if not txt_path:
        return set()
    if not os.path.exists(txt_path):
        logger.warning(f"ID list file not found: {txt_path}. Fallback to random sampling.")
        return set()
    ids: Set[str] = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.add(s)
    return ids


# --------------------- 新增分层采样函数 --------------------- #
def stratified_subsample_by_ids(
    records: List[Dict[str, Any]],
    n: int,
    wrong_ratio: float,
    correct_ids: Set[str],
    wrong_ids: Set[str],
    seed: int,
    split_name: str,
) -> List[Dict[str, Any]]:  # CHANGED/NEW
    """
    基于给定的答对/答错id集合，按wrong_ratio进行分层随机抽样，总量为n。
    回退策略：
      - 若某一类不足，另一类补齐。
      - 若两类集合都为空，则退回到纯随机抽样。
    """
    rnd = random.Random(seed)
    # 基于记录内的id进行交集，避免id清单包含不在本split内的样本
    id_to_rec: Dict[str, Dict[str, Any]] = {str(r.get("id")): r for r in records if r.get("id") is not None}
    correct_pool = [id_to_rec[i] for i in correct_ids]
    wrong_pool = [id_to_rec[i] for i in wrong_ids]

    if len(correct_pool) == 0 and len(wrong_pool) == 0:
        logger.warning(f"[{split_name}] Both correct_ids and wrong_ids empty or unmatched. Fallback to random sampling.")
        # 退回纯随机
        idx = list(range(len(records)))
        rnd.shuffle(idx)
        pick = idx[: min(n, len(idx))]
        return [records[i] for i in pick]

    # 目标数量
    n = min(n, len(records))
    target_wrong = int(round(n * wrong_ratio))
    target_correct = n - target_wrong

    # 打乱池
    rnd.shuffle(correct_pool)
    rnd.shuffle(wrong_pool)

    sel_wrong = wrong_pool[: min(target_wrong, len(wrong_pool))]
    sel_correct = correct_pool[: min(target_correct, len(correct_pool))]

    # 若总量不足，进行补齐
    deficit = n - (len(sel_wrong) + len(sel_correct))
    if deficit > 0:
        """ # 先从未用过的wrong补
        remaining_wrong = [r for r in wrong_pool if r not in sel_wrong]
        remaining_correct = [r for r in correct_pool if r not in sel_correct]
        # 交替补齐，优先补样本更充足的一侧
        pools = sorted(
            [("wrong", remaining_wrong), ("correct", remaining_correct)],
            key=lambda x: len(x[1]),
            reverse=True,
        )
        for _, pool in pools:
            if deficit <= 0:
                break
            take = min(deficit, len(pool))
            if take > 0:
                sel = pool[:take]
                if pool is remaining_wrong:
                    sel_wrong.extend(sel)
                else:
                    sel_correct.extend(sel)
                deficit -= take """
        raise ValueError(111)

    selected = sel_wrong + sel_correct
    if len(selected) < n:
        logger.warning(f"[{split_name}] Unable to reach requested sample size {n}. Got {len(selected)}.")
    # 打乱最终顺序，保持随机性
    rnd.shuffle(selected)
    return selected[:n]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", default="/aifs4su/xmed/zahuai/datasets/datasets--rajpurkarlab--ReXVQA/snapshots/55f531580174d43117c467733a2e4de2c3a63915/metadata/train_vqa_data.json", help="Path to raw train json/jsonl")
    parser.add_argument("--test_json", default="/aifs4su/xmed/zahuai/datasets/datasets--rajpurkarlab--ReXVQA/snapshots/55f531580174d43117c467733a2e4de2c3a63915/metadata/test_vqa_data.json", help="Path to raw test json/jsonl")
    parser.add_argument("--out_dir", default="/home/xmed/zahuai/RL-Factory/ReXVQA", help="Output directory")
    parser.add_argument("--train_sample", type=int, default=2000)
    parser.add_argument("--test_sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--wrong_ratio", type=float, default=0.3, help="Proportion of wrong-answer samples in the subsampled splits, in [0,1].")
    parser.add_argument("--train_correct_ids", type=str, default="/aifs4su/xmed/zahuai/RL-Factory/baselines/result_analysis/train_correct_ids.txt", help="Path to txt of correct IDs for train split.") 
    parser.add_argument("--train_wrong_ids", type=str, default="/aifs4su/xmed/zahuai/RL-Factory/baselines/result_analysis/train_wrong_ids.txt", help="Path to txt of wrong IDs for train split.")  
    parser.add_argument("--test_correct_ids", type=str, default="/aifs4su/xmed/zahuai/RL-Factory/baselines/result_analysis/test_correct_ids.txt", help="Path to txt of correct IDs for test split.")
    parser.add_argument("--test_wrong_ids", type=str, default="/aifs4su/xmed/zahuai/RL-Factory/baselines/result_analysis/test_wrong_ids.txt", help="Path to txt of wrong IDs for test split.") 

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    logger.info("Loading datasets ...")
    train_ds = read_any_json_dataset(args.train_json)
    test_ds = read_any_json_dataset(args.test_json)
    logger.info(f"Raw sizes - train: {len(train_ds)}, test: {len(test_ds)}")

    # 读取id清单
    train_correct_ids = read_id_list(args.train_correct_ids)
    train_wrong_ids = read_id_list(args.train_wrong_ids)
    test_correct_ids = read_id_list(args.test_correct_ids)
    test_wrong_ids = read_id_list(args.test_wrong_ids)

    logger.info("Subsampling ...")

    # 若提供了对应id清单，使用分层抽样；否则退回原随机抽样  # CHANGED/NEW
    if train_correct_ids or train_wrong_ids:
        logger.info(f"Using stratified sampling for train with wrong_ratio={args.wrong_ratio}")
        train_ds = stratified_subsample_by_ids(
            records=train_ds,
            n=args.train_sample,
            wrong_ratio=args.wrong_ratio,
            correct_ids=train_correct_ids,
            wrong_ids=train_wrong_ids,
            seed=args.seed,
            split_name="train",
        )
    else:
        train_ds = subsample_dataset(train_ds, args.train_sample, args.seed)

    if test_correct_ids or test_wrong_ids:
        logger.info(f"Using stratified sampling for test with wrong_ratio={args.wrong_ratio}")
        test_ds = stratified_subsample_by_ids(
            records=test_ds,
            n=args.test_sample,
            wrong_ratio=args.wrong_ratio,
            correct_ids=test_correct_ids,
            wrong_ids=test_wrong_ids,
            seed=args.seed,
            split_name="test",
        )
    else:
        test_ds = subsample_dataset(test_ds, args.test_sample, args.seed)
        
    logger.info(f"Sampled sizes - train: {len(train_ds)}, test: {len(test_ds)}")

    # map
    logger.info("Mapping train ...")
    """ train_proc = train_ds.map(lambda ex: map_example(ex, "train"), num_proc=8)
    train_proc = train_proc.filter(lambda x: x is not None) """
    train_proc = map_records(train_ds, "train")
    logger.info("Mapping test ...")
    """ test_proc = test_ds.map(lambda ex: map_example(ex, "test"), num_proc=8)
    test_proc = test_proc.filter(lambda x: x is not None) """
    test_proc = map_records(test_ds, "test")

    # Save
    train_out = os.path.join(args.out_dir, "train_processed.jsonl")
    test_out = os.path.join(args.out_dir, "test_processed.jsonl")
    logger.info(f"Saving to {train_out} and {test_out}")
    # custom writer to avoid escaped slash
    #def write_jsonl(ds: datasets.Dataset, out_path: str):
    def write_jsonl(records: List[Dict[str, Any]], out_path: str):
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in records: #ds
                line = json.dumps(ex, ensure_ascii=False).replace("\\/", "/")
                f.write(line + "\n")

    write_jsonl(train_proc, train_out)
    write_jsonl(test_proc, test_out)

    logger.info("Done.")

if __name__ == "__main__":
    main()