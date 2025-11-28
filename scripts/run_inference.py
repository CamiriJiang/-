from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger("run_inference")


@dataclass
class InferenceConfig:
    # 使用训练脚本输出的目录
    model_dir: Path = Path("artifacts/seq2seq_opus_mt_en_zh")

    # 数据来源（可选）
    processed_dir: Path = Path("processed")
    test_file: str = "test.jsonl"
    sample_size: int = 5

    # 生成参数
    max_source_length: int = 96
    max_new_tokens: int = 80
    num_beams: int = 4
    temperature: float = 1.0
    top_k: int = 50
    do_sample: bool = False  # 可改 True 支持随机生成


def load_model_and_tokenizer(cfg: InferenceConfig):
    """加载真正的 Seq2Seq 模型并优化性能"""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_dir)

    # 推理性能优化
    model.config.use_cache = True  # KV cache → 生成更快

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # PyTorch 2.0 编译加速
    if torch.__version__.startswith("2"):
        try:
            model = torch.compile(model)
            LOGGER.info("已启用 torch.compile 模型加速")
        except Exception:
            LOGGER.warning("compile() 加速失败，跳过")

    LOGGER.info(f"模型已加载，设备：{device}")
    return model, tokenizer, device


def read_test_samples(cfg: InferenceConfig, limit: int) -> List[dict]:
    """从 processed/test.jsonl 中读取测试样例"""
    path = cfg.processed_dir / cfg.test_file
    if not path.exists():
        raise FileNotFoundError(f"测试文件不存在：{path}")

    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    return samples


def translate_batch(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device,
    texts: List[str],
    cfg: InferenceConfig,
):
    """批量翻译 + FP16 加速"""
    encoded = tokenizer(
        texts,
        max_length=cfg.max_source_length,
        truncation=True,
        padding="longest",      # GPU 最优 padding
        pad_to_multiple_of=8,   # TensorCore 加速
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # 混合精度推理
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        outputs = model.generate(
            **encoded,
            max_new_tokens=cfg.max_new_tokens,
            num_beams=cfg.num_beams,
            do_sample=cfg.do_sample,
            top_k=cfg.top_k,
            temperature=cfg.temperature,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main(cfg: InferenceConfig = InferenceConfig()):
    """批量推理 + 手动输入模式"""

    model, tokenizer, device = load_model_and_tokenizer(cfg)

    # ====== 先跑一批测试样例 ======
    samples = read_test_samples(cfg, cfg.sample_size)
    srcs = [s["source"] for s in samples]
    tgts = [s["target"] for s in samples]

    preds = translate_batch(model, tokenizer, device, srcs, cfg)

    for s, t, p in zip(srcs, tgts, preds):
        LOGGER.info(f"源句: {s}")
        LOGGER.info(f"参考: {t}")
        LOGGER.info(f"预测: {p}")
        LOGGER.info("-" * 40)

    # ====== 手动输入模式 ======
    print("\n======== 手动输入测试模式（输入 q 退出） ========\n")
    while True:
        text = input("请输入句子： ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            print("已退出。")
            break

        pred = translate_batch(model, tokenizer, device, [text], cfg)[0]
        print(f"翻译结果：{pred}\n")


if __name__ == "__main__":
    main()
