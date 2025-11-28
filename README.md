# 机器翻译项目（Seq2Seq + Attention + BERT）

本项目基于 `AI Challenger 2017` 英中翻译数据集，构建一个使用 **BERT 双向编码器-解码器**（本质为带注意力机制的 Seq2Seq）模型的机器翻译训练与推理流程。项目包含数据预处理、模型训练、推理测试与评估四大脚本，参数通过脚本内的配置类集中管理，避免命令行参数带来的复杂度。

## 目录结构

- `data/AIchallenger2017.zip`：原始数据集（已提供）
- `processed/`：数据预处理输出（运行 `prepare_data.py` 后生成）
- `artifacts/seq2seq_bert/`：训练完成的模型与指标（运行 `train_seq2seq.py` 后生成）
- `scripts/`
  - `prepare_data.py`：数据解析与清洗
  - `train_seq2seq.py`：BERT 编码器-解码器模型训练
  - `run_inference.py`：推理与样例翻译测试
  - `evaluate_model.py`：BLEU 等指标评估
- `requirements.txt`：所需 Python 依赖

## 环境准备

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

> **提示**：首次安装 `torch` 时请根据机器的 CUDA/CPU 情况参考官方说明选择合适的版本。

## 使用步骤

1. **预处理数据**
   ```bash
   python scripts/prepare_data.py
   ```
   - 解析 `zip` 内的训练/验证文件，清洗、对齐并输出 `processed/train.jsonl`、`processed/valid.jsonl`、`processed/test.jsonl` 及 `meta.json`。

2. **训练模型**
   ```bash
   python scripts/train_seq2seq.py
   ```
   - 使用 `bert-base-multilingual-cased` 初始化 `EncoderDecoderModel`。
   - 通过 `Seq2SeqTrainer` 训练，默认保存到 `artifacts/seq2seq_bert`。

3. **推理测试**
   ```bash
   python scripts/run_inference.py
   ```
   - 加载上一步生成的模型，对自定义或测试集样例进行翻译并输出结果。

4. **模型评估**
   ```bash
   python scripts/evaluate_model.py
   ```
   - 在 `processed/test.jsonl` 上计算 BLEU，并输出预测示例以辅助人工检查。

## 参数与二次开发

- 每个脚本顶部都定义了 `@dataclass` 配置，修改类字段即可更改路径、训练规模、生成策略等重要参数。
- 核心函数均提供注释与日志，便于排查问题或扩展其他模型结构（如更换为 `mT5`、引入知识蒸馏等）。

## 后续工作建议

- 增加更多正则化与数据增强策略，提升长句翻译质量。
- 尝试混合精度/梯度检查点以降低显存占用。
- 引入多参考评估（如 chrF、COMET）以获得更全面的指标反馈。


