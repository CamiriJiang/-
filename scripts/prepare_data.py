"""
数据预处理脚本：
- 解析 AI Challenger 2017 压缩包中的训练/验证集
- 清洗、对齐双语句对
- 输出 JSONL 供后续训练、评估使用
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from zipfile import ZipFile

# 配置日志格式和级别
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
# 创建日志记录器实例
LOGGER = logging.getLogger("prepare_data")

# 编译正则表达式模式，用于匹配SGM文件中的<seg>标签内容
# 支持跨行匹配和忽略大小写
SEG_PATTERN = re.compile(r"<seg[^>]*>(.*?)</seg>", re.S | re.I)


@dataclass
class PrepareConfig:
    """集中管理所有可调参数，避免命令行依赖。"""

    zip_path: Path = Path(r'../data/AIchallenger2017.zip')
    output_dir: Path = Path("processed")

    train_en_path: str = (
        "AIchallenger2017/ai_challenger_translation_train_20170904/"
        "translation_train_data_20170904/train.en"
    )
    train_zh_path: str = (
        "AIchallenger2017/ai_challenger_translation_train_20170904/"
        "translation_train_data_20170904/train.zh"
    )
    valid_en_path: str = (
        "AIchallenger2017/ai_challenger_translation_validation_20170912/"
        "translation_validation_20170912/valid.en-zh.en.sgm"
    )
    valid_zh_path: str = (
        "AIchallenger2017/ai_challenger_translation_validation_20170912/"
        "translation_validation_20170912/valid.en-zh.zh.sgm"
    )

    # 数据体量 & 清洗策略
    min_char_len: int = 1
    max_char_len: int = 200
    train_sample_limit: int | None = 200_000
    valid_sample_limit: int | None = 5_000
    validation_split_ratio: float = 0.5  # 将官方验证集一分为二：valid/test

    seed: int = 42


def _ensure_paths(cfg: PrepareConfig) -> None:
    """确保输入压缩包存在并创建输出目录。

    参数:
        cfg: 提供数据源与输出位置的配置。
    """
    if not cfg.zip_path.exists():
        raise FileNotFoundError(f"找不到数据集压缩包：{cfg.zip_path}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)


def _read_zip_text(zf: ZipFile, inner_path: str) -> str:
    """读取压缩包内指定文件并以 UTF-8 解码成字符串。

    参数:
        zf: 已打开的 ZipFile 对象。
        inner_path: 压缩包内的相对路径。
    """
    with zf.open(inner_path, "r") as handle:
        return handle.read().decode("utf-8")


def _read_zip_lines(zf: ZipFile, inner_path: str) -> List[str]:
    """读取压缩包文本文件并按行返回清洗后的内容。

    参数:
        zf: 已打开的 ZipFile 对象。
        inner_path: 压缩包内的相对路径。
    """
    content = _read_zip_text(zf, inner_path)
    lines = [clean_text(line) for line in content.splitlines()]
    return [line for line in lines if line]


def clean_text(text: str) -> str:
    """去除多余空白并转义 HTML，保证句子干净。

    参数:
        text: 原始文本。
    """
    text = unescape(text.strip())
    text = re.sub(r"\s+", " ", text)
    return text


def parse_sgm_segments(content: str) -> List[str]:
    """解析 SGM 文件中的 <seg> 标签，提取句子。

    参数:
        content: SGM 文件完整字符串。
    """
    return [clean_text(seg) for seg in SEG_PATTERN.findall(content)]


def build_pairs(
    src_lines: Sequence[str],
    tgt_lines: Sequence[str],
    cfg: PrepareConfig,
) -> List[Dict[str, str]]:
    """将源/目标句子对齐为字典列表并按长度过滤。

    参数:
        src_lines: 英文句子序列。
        tgt_lines: 中文句子序列。
        cfg: 控制长度阈值的配置。
    """
    pairs: List[Dict[str, str]] = []
    for src, tgt in zip(src_lines, tgt_lines):
        if not (cfg.min_char_len <= len(src) <= cfg.max_char_len):
            continue
        if not (cfg.min_char_len <= len(tgt) <= cfg.max_char_len):
            continue
        pairs.append({"source": src, "target": tgt})
    return pairs


def limit_pairs(
    pairs: List[Dict[str, str]],
    limit: int | None,
) -> List[Dict[str, str]]:
    """用于抽样，按给定上限截断句对集合。

    参数:
        pairs: 原始句对列表。
        limit: 最大保留数量，None 表示不过滤。
    """
    if limit is None or len(pairs) <= limit:
        return pairs
    return pairs[:limit]


def write_jsonl(path: Path, pairs: Sequence[Dict[str, str]]) -> None:
    """将句对序列写入 JSONL 文件。

    参数:
        path: 输出文件路径。
        pairs: 包含 source/target 的字典序列。
    """
    with path.open("w", encoding="utf-8") as fp:
        for record in pairs:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    LOGGER.info("写出 %s，共 %d 条样本", path, len(pairs))


def summarize_counts(train: Sequence, valid: Sequence, test: Sequence) -> Dict[str, int]:
    """汇总各数据 split 的样本量，方便记录/调试。

    参数:
        train: 训练数据集合。
        valid: 验证数据集合。
        test: 测试数据集合。
    """
    return {
        "train_samples": len(train),
        "valid_samples": len(valid),
        "test_samples": len(test),
    }


def main(cfg: PrepareConfig = PrepareConfig()) -> None:
    """脚本主入口：完成数据读取、清洗、划分与落盘。

    参数:
        cfg: 控制路径、抽样规模及随机种子的配置。
    """
    _ensure_paths(cfg)
    random.seed(cfg.seed)

    with ZipFile(cfg.zip_path) as zf:
        LOGGER.info("载入训练集...")
        train_en = _read_zip_lines(zf, cfg.train_en_path)
        train_zh = _read_zip_lines(zf, cfg.train_zh_path)
        train_pairs = build_pairs(train_en, train_zh, cfg)
        random.shuffle(train_pairs)
        train_pairs = limit_pairs(train_pairs, cfg.train_sample_limit)

        LOGGER.info("载入验证集（将用于 valid/test）...")
        valid_en = parse_sgm_segments(_read_zip_text(zf, cfg.valid_en_path))
        valid_zh = parse_sgm_segments(_read_zip_text(zf, cfg.valid_zh_path))
        valid_pairs_full = build_pairs(valid_en, valid_zh, cfg)
        valid_pairs_full = limit_pairs(valid_pairs_full, cfg.valid_sample_limit)

    split_idx = int(len(valid_pairs_full) * cfg.validation_split_ratio)
    split_idx = max(1, min(split_idx, len(valid_pairs_full) - 1)) if len(valid_pairs_full) > 1 else len(valid_pairs_full)
    valid_pairs = valid_pairs_full[:split_idx]
    test_pairs = valid_pairs_full[split_idx:] or valid_pairs_full

    outputs = {
        "train": cfg.output_dir / "train.jsonl",
        "valid": cfg.output_dir / "valid.jsonl",
        "test": cfg.output_dir / "test.jsonl",
        "meta": cfg.output_dir / "meta.json",
    }

    write_jsonl(outputs["train"], train_pairs)
    write_jsonl(outputs["valid"], valid_pairs)
    write_jsonl(outputs["test"], test_pairs)

    meta = summarize_counts(train_pairs, valid_pairs, test_pairs)
    outputs["meta"].write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("数据准备完成：%s", meta)


if __name__ == "__main__":
    main()


