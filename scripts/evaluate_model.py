"""
评估脚本：用于评估单向翻译模型（英文 → 中文）。
使用 AutoModelForSeq2SeqLM + SacreBLEU。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from sacrebleu import corpus_bleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger("evaluate_seq2seq")


@dataclass
class EvalConfig:
    model_dir: Path = Path("artifacts/seq2seq_opus_mt_en_zh")  # 训练脚本输出目录
    processed_dir: Path = Path("processed")
    test_file: str = "test.jsonl"

    batch_size: int = 8
    max_source_length: int = 96
    max_new_tokens: int = 96
    num_beams: int = 4


def load_test_samples(cfg: EvalConfig) -> List[dict]:
    """加载 test.jsonl"""
    path = cfg.processed_dir / cfg.test_file
    if not path.exists():
        raise FileNotFoundError(f"测试文件不存在: {path}")

    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    LOGGER.info(f"读取测试集 {len(samples)} 条")
    return samples


def load_model_and_tokenizer(cfg: EvalConfig):
    """加载模型 + tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    LOGGER.info(f"模型加载完成，设备：{device}")
    return model, tokenizer, device


def translate_batch(
    model, tokenizer, device, texts: List[str], cfg: EvalConfig
):
    """批量推理 + FP16 加速"""
    encoded = tokenizer(
        texts,
        max_length=cfg.max_source_length,
        truncation=True,
        padding="longest",
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # 混合精度推理
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        outputs = model.generate(
            **encoded,
            max_new_tokens=cfg.max_new_tokens,
            num_beams=cfg.num_beams,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def chunk(lst: List, n: int):
    """把 list 分成 batch 块"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main(cfg: EvalConfig = EvalConfig()):
    samples = load_test_samples(cfg)
    model, tokenizer, device = load_model_and_tokenizer(cfg)

    predictions = []
    references = []

    for batch in chunk(samples, cfg.batch_size):
        srcs = [s["source"] for s in batch]
        tgts = [s["target"] for s in batch]

        preds = translate_batch(model, tokenizer, device, srcs, cfg)

        predictions.extend(preds)
        references.extend(tgts)

    # ===== BLEU 计算 =====
    bleu = corpus_bleu(predictions, [references]).score
    avg_len = np.mean([len(p.split()) for p in predictions])

    metrics = {
        "bleu": float(bleu),
        "avg_pred_tokens": float(avg_len),
        "total_samples": len(samples),
    }

    # 保存
    save_path = cfg.model_dir / "eval_metrics.json"
    save_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    LOGGER.info(f"评估完成: {metrics}")

    # 打印样例
    LOGGER.info("\n===== 样例输出 =====")
    for i in range(min(3, len(samples))):
        LOGGER.info(f"源句: {samples[i]['source']}")
        LOGGER.info(f"参考: {samples[i]['target']}")
        LOGGER.info(f"预测: {predictions[i]}")
        LOGGER.info("-------------------------")


if __name__ == "__main__":
    main()
