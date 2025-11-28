"""
训练脚本：使用真正的 Seq2Seq 模型（如 MarianMT / T5）完成英中翻译任务。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from datasets import DatasetDict, load_dataset
from sacrebleu import corpus_bleu
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# 关闭 tokenizer 的并行 warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger("train_seq2seq")


@dataclass
class TrainConfig:
    # 数据相关路径配置
    processed_dir: Path = Path("processed")  # 预处理数据存放目录
    train_file: str = "train.jsonl"          # 训练数据文件名
    valid_file: str = "valid.jsonl"          # 验证数据文件名

    # ✅ 使用真正的 Seq2Seq 模型（默认：英→中）
    # 你可以换成例如 "t5-small" / "facebook/mbart-large-50-many-to-many-mmt" 等
    model_name: str = "Helsinki-NLP/opus-mt-en-zh"
    output_dir: Path = Path("artifacts/seq2seq_opus_mt_en_zh")

    # 序列长度配置
    max_source_length: int = 96      # 源文本（英语）最大长度
    max_target_length: int = 96      # 目标文本（中文）最大长度
    generation_max_length: int = 96  # 生成文本最大长度

    # 训练超参数配置
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    num_train_epochs: float = 1.0
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    fp16: bool = True  # 3060Ti 可以放心开混合精度

    # 数据子集配置（用于快速测试）
    train_subset: int | None = 2000    # None 表示用全部训练集
    valid_subset: int | None = 200     # None 表示用全部验证集
    seed: int = 42


def load_parallel_dataset(cfg: TrainConfig) -> DatasetDict:
    """加载预处理好的 JSONL 数据，并根据配置抽样。"""

    data_files = {
        "train": str(cfg.processed_dir / cfg.train_file),
        "validation": str(cfg.processed_dir / cfg.valid_file),
    }
    dataset = load_dataset("json", data_files=data_files)

    def _maybe_slice(split: str, limit: int | None):
        if limit is None:
            return dataset[split]
        return dataset[split].select(range(min(limit, len(dataset[split]))))

    dataset["train"] = _maybe_slice("train", cfg.train_subset)
    dataset["validation"] = _maybe_slice("validation", cfg.valid_subset)
    return dataset


def prepare_tokenizer(cfg: TrainConfig):
    """构建分词器。Seq2Seq 模型自带正确的 bos/eos/pad 配置。"""

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    # 保险起见，确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_model(cfg: TrainConfig, tokenizer):
    """加载 Seq2Seq 模型（已内置 decoder 与注意力机制）。"""

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    # 将生成相关配置写入模型（Trainer 的 predict_with_generate 会用到）
    model.config.max_length = cfg.generation_max_length
    model.config.num_beams = 4
    model.config.length_penalty = 1.0
    model.config.no_repeat_ngram_size = 3

    # 有些模型（如 mbart）需要设置 src_lang / tgt_lang，这里用 Marian 就不需要
    return model


def tokenize_batch_factory(tokenizer, cfg: TrainConfig):
    """返回批量分词函数，兼顾源句与目标句。"""

    def _tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        # 源语言：英文
        model_inputs = tokenizer(
            batch["source"],
            max_length=cfg.max_source_length,
            truncation=True,
            padding="max_length",
        )
        # 目标语言：中文（注意 text_target）
        labels = tokenizer(
            text_target=batch["target"],
            max_length=cfg.max_target_length,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return _tokenize


def compute_metrics_factory(tokenizer):
    """构建评估函数，供 Trainer 回调计算 BLEU。"""

    def _compute(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # 解码预测序列
        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        # 将 label 中的 -100 换回 pad_token_id 方便解码
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        bleu = corpus_bleu(decoded_preds, [decoded_labels]).score
        pred_lens = [len(p.split()) for p in decoded_preds]
        return {
            "bleu": bleu,
            "avg_pred_tokens": float(np.mean(pred_lens)) if pred_lens else 0.0,
        }

    return _compute


def save_metrics(output_dir: Path, metrics: Dict[str, Any]) -> None:
    """将训练/验证指标写入 JSON 文件。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("训练指标已保存：%s", metrics_path)


def main(cfg: TrainConfig = TrainConfig()) -> None:
    """脚本主入口：加载数据、训练模型并保存成果。"""

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    dataset = load_parallel_dataset(cfg)
    tokenizer = prepare_tokenizer(cfg)
    model = prepare_model(cfg, tokenizer)

    tokenized_dataset = dataset.map(
        tokenize_batch_factory(tokenizer, cfg),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps",              # 你当前 transformers 版本用 eval_strategy
        predict_with_generate=True,
        generation_max_length=cfg.generation_max_length,
        fp16=cfg.fp16,
        report_to=[],                       # 不上报到 wandb 等
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        seed=cfg.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_factory(tokenizer),
    )

    LOGGER.info(
        "开始训练，数据量：train=%d, valid=%d",
        len(dataset["train"]),
        len(dataset["validation"]),
    )
    train_result = trainer.train()
    trainer.save_model(cfg.output_dir)

    metrics = {**train_result.metrics, **trainer.evaluate()}
    save_metrics(cfg.output_dir, metrics)


if __name__ == "__main__":
    main()
