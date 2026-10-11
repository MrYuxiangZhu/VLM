# Qwen 视频目标检测与分类微调框架

这是一个基于 Qwen3-VL（用户所说的“Qwen3.7”若为内部模型，请把配置中的模型路径替换为实际 checkpoint）的完整 LoRA/QLoRA 风格训练骨架。输入视频，输出严格 JSON：视频级分类、目标类别、置信度和归一化边界框。

## 1. 功能和设计

- 使用 Qwen 的原生视频输入能力，不把视频预先拼成图片。
- 使用 LoRA，只训练语言模型适配层，降低显存和数据量要求。
- 统一输出格式：`video_class` + `detections`。
- 支持训练、单视频推理、批量评估和数据集切分。
- 检测框采用 `[x1, y1, x2, y2]`，相对视频宽高归一化到 `[0, 1]`，便于跨分辨率训练。
- 默认采用视频级目标检测：模型从采样帧中汇总目标。如果需要逐帧检测，应把标注扩展为 `frame_index`，并在输出协议中加入帧号。

## 2. 数据集与模块化架构

训练数据统一抽象为 `src/data/datasets.py` 中的 `VisionDataset`。当前内置 `jsonl` 和 `coco` 两种适配方式；新增数据集时，只需实现相同的 `__getitem__` 消息协议，再在 `build_dataset` 注册即可。训练、数据、日志和评估相互独立，后续可以追加视频、图片或自定义标注源。

本次已接入 `/home/yuxiangzhu/volume/animal_det/data/coco` 下的 COCO `instances_train2017.json` 和 `instances_val2017.json`。COCO 是图片检测数据，因此会以 Qwen-VL 的 image 输入训练；视频数据仍使用 JSONL 的 `media_type: video`。

先转换 COCO：

```bash
python3 scripts/convert_coco.py \\
  --annotation /home/yuxiangzhu/volume/animal_det/data/coco/annotations/instances_train2017.json \\
  --image-root /home/yuxiangzhu/volume/animal_det/data/coco/train2017 \\
  --output data/coco_train.jsonl
python3 scripts/convert_coco.py \\
  --annotation /home/yuxiangzhu/volume/animal_det/data/coco/annotations/instances_val2017.json \\
  --image-root /home/yuxiangzhu/volume/animal_det/data/coco/val2017 \\
  --output data/coco_val.jsonl
```

如果要追加多个数据集，配置可以写成列表，训练器会自动使用 `ConcatDataset`：

```yaml
train_dataset:
  - {type: coco, annotations: data/coco_train.jsonl, media_root: /path/train2017}
  - {type: jsonl, annotations: data/custom.jsonl, media_root: /path/custom, media_type: video}
```

日志写入 `outputs/logs/run.log`，Transformers 的 loss、learning rate、eval loss 同时写入 `log_dir`。日志统一使用时间、级别、模块名格式。

## 3. 环境

建议 Linux + NVIDIA GPU，至少 24 GB 显存用于 7B/8B 的 LoRA；显存不足时降低 `video_fps`、缩短视频或使用 4-bit 量化。安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 使用了 Qwen 视频处理、Transformers、PEFT 和 TRL。模型名称可能随 Transformers 版本变化：当前配置默认 `Qwen/Qwen3-VL-8B-Instruct`，若你的“Qwen3.7”是私有或其他命名模型，只需要修改 `configs/train.yaml` 的 `model_name_or_path`。

## 3. 数据格式

每行一个 JSONL 样本，视频路径可以是绝对路径，也可以相对于数据文件所在工作目录：

```json
{"video":"videos/0001.mp4","instruction":"检测所有车辆，并在 [car, truck, bus] 中选择视频级类别。","output":{"video_class":"car","detections":[{"label":"car","score":0.95,"bbox":[0.12,0.18,0.56,0.88]}]}}
```

建议：

1. 训练集、验证集、测试集按视频划分，不能把同一视频的相邻片段跨集合，否则会造成数据泄漏。
2. 框坐标统一转换为归一化坐标；标注越界、空框和类别名称要在预处理时清理。
3. 每个类别至少准备足够的正负样本，分类类别写进 instruction，避免训练/推理类别集合不一致。
4. 若同一目标在多个时间点出现，当前格式只监督聚合框；如果需要跟踪，增加 `track_id` 和 `frame_index` 字段。

用已有 JSONL 切分：

```bash
python scripts/split_dataset.py --input all.jsonl --output-dir data
```

然后按 JSONL 文件的相对路径放置视频（例如 `video` 写成 `videos/0001.mp4` 时，文件位于 `data/videos/0001.mp4`），并把 `data/train.jsonl` 和 `data/val.jsonl` 中的路径改好。`data/example.jsonl` 可作为模板。

## 4. Conda 环境搭建

使用默认 CUDA PyTorch 源创建并配置环境：

```bash
bash scripts/setup_conda.sh
```

默认环境名为 `vlm-qwen`，Python 版本为 3.11。环境名和 Python 版本可以通过环境变量修改：

```bash
ENV_NAME=qwen-video PYTHON_VERSION=3.11 bash scripts/setup_conda.sh
```

脚本会完成：

- 创建或复用 Conda 环境
- 安装 CUDA 版 PyTorch、TorchVision 和 TorchAudio
- 安装 `requirements.txt`
- 执行 Python 语法检查

如果使用 CPU 环境：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \\
bash scripts/setup_conda.sh
```

如果使用其他 CUDA 版本，可以覆盖 PyTorch wheel 源：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \\
bash scripts/setup_conda.sh
```

安装完成后：

```bash
conda activate vlm-qwen
```

## 5. Shell 快捷脚本

所有脚本都会自动切换到项目根目录，并支持 `PYTHON_BIN`、`GPU_ID`、`MODEL` 等环境变量。

训练：

```bash
bash scripts/train.sh configs/train.yaml
```

单图片推理：

```bash
bash scripts/infer.sh data/example.jpg outputs/prediction.json
```

视频推理：

```bash
bash scripts/video_infer.sh data/example.mp4 outputs/video_prediction.json 2
```

验证集评估：

```bash
bash scripts/evaluate.sh data/coco_val.jsonl /home/yuxiangzhu/volume/animal_det/data/coco/val2017
```

指定模型或 GPU：

```bash
MODEL=outputs/my-model GPU_ID=1 bash scripts/train.sh
MODEL=outputs/my-model bash scripts/infer.sh data/example.jpg
```

脚本文件：

```text
scripts/train.sh
scripts/infer.sh
scripts/video_infer.sh
scripts/evaluate.sh
```

## 6. 微调模式完整说明

项目通过 `src/model/factory.py` 统一创建模型，通过 `scripts/finetune.sh` 选择训练模式。脚本调用格式如下：

```bash
bash scripts/finetune.sh <mode> [config_path] [output_dir]
```

参数说明：

| 参数 | 是否必填 | 默认值 | 说明 |
|---|---|---|---|
| `mode` | 否 | `lora` | 可选 `lora`、`qlora`、`dora`、`full` |
| `config_path` | 否 | `configs/train.yaml` | 训练配置文件 |
| `output_dir` | 否 | 配置中的 `output_dir` | 覆盖模型和 checkpoint 输出目录 |

命令行模式会覆盖配置文件中的 `finetune_mode`。第三个参数会覆盖配置文件中的 `output_dir`。

### 6.1 LoRA

LoRA 冻结基础模型权重，只在指定线性层上训练低秩适配器。它通常是效果、显存、训练速度之间最均衡的方案，建议作为首个实验基线。

```bash
GPU_ID=0 bash scripts/finetune.sh \
  lora \
  configs/train.yaml \
  outputs/qwen3-coco-lora
```

适用场景：

- 单卡 24 GB 或更高显存；
- 需要快速验证数据质量和输出格式；
- 希望保留基础模型，并为不同任务分别保存小型 adapter；
- COCO 图片检测、短视频理解等常规任务。

相关配置：

```yaml
finetune_mode: lora
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
lora_bias: none
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

`lora_r` 越大，可训练参数越多，表达能力和显存占用也越高。建议从 16、32 或 64 开始。`lora_alpha` 常设为 `r` 或 `2r`。数据量较小时可将 `lora_dropout` 设置为 0.05～0.1。

LoRA 输出主要是 adapter 权重和配置，不是独立的完整基础模型。推理时必须能够访问原始基础模型。

### 6.2 QLoRA

QLoRA 以 4-bit 形式加载冻结的基础模型，并训练 LoRA adapter。当前实现使用 NF4 和双重量化：

```yaml
finetune_mode: qlora
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true
```

启动命令：

```bash
GPU_ID=0 bash scripts/finetune.sh \
  qlora \
  configs/train.yaml \
  outputs/qwen3-coco-qlora
```

适用场景：

- GPU 显存有限；
- 16～24 GB 单卡上尝试 7B/8B 模型；
- 视频帧数较少、分辨率经过限制的微调；
- 可以接受量化加载带来的额外计算开销。

QLoRA 要求：

- NVIDIA CUDA GPU；
- Linux 环境；
- 已安装 `bitsandbytes`；
- PyTorch CUDA 版本与 NVIDIA 驱动兼容。

检查环境：

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
python3 -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

如果出现 CUDA OOM，优先降低：

```yaml
max_seq_length: 1024
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 16
```

视频训练还应降低数据配置中的 `fps`，并提前限制视频长度和分辨率。

### 6.3 DoRA

DoRA 将权重更新分解为方向和幅度两部分，仍然通过 PEFT adapter 训练。当前实现使用：

```python
use_dora=True
```

启动命令：

```bash
GPU_ID=0 bash scripts/finetune.sh \
  dora \
  configs/train.yaml \
  outputs/qwen3-coco-dora
```

DoRA 通常比标准 LoRA 占用更多显存和计算资源，但在部分任务上能获得更好的效果。建议在 LoRA 基线稳定后，再使用完全相同的数据划分和超参数对比 DoRA。

DoRA 当前不是 4-bit QDoRA；它按 BF16/FP16 加载基础模型。如果需要 QDoRA，需要进一步组合量化加载和 `use_dora=True`，并验证所用 PEFT、bitsandbytes 和模型结构的兼容性。

### 6.4 Full SFT

Full SFT 不创建 adapter，模型全部参数参与训练：

```bash
GPU_ID=0 bash scripts/finetune.sh \
  full \
  configs/train.yaml \
  outputs/qwen3-coco-full
```

适用场景：

- 数据量足够大且标注质量高；
- 需要显著改变模型能力；
- 有多卡高显存 GPU；
- 能接受较大的 checkpoint、优化器状态和训练成本。

对于 7B/8B 模型，Full SFT 的显存不仅包含模型权重，还包括梯度、优化器状态和激活值。单卡 24～48 GB 通常不足，建议使用多张 A100/H100 80 GB，并进一步接入 DeepSpeed ZeRO-3 或 FSDP。当前脚本是基础 Trainer 入口，不会自动启用 ZeRO/FSDP。

### 6.5 模式对比

| 模式 | 基础模型加载 | 训练参数 | 显存需求 | 输出形式 | 推荐用途 |
|---|---|---:|---:|---|---|
| `lora` | BF16/FP16 | 少量 adapter | 中 | adapter | 默认基线、24 GB+ 单卡 |
| `qlora` | 4-bit NF4 | 少量 adapter | 最低 | adapter | 16～24 GB 单卡 |
| `dora` | BF16/FP16 | adapter | 中高 | adapter | LoRA 效果对比 |
| `full` | BF16/FP16 | 全部参数 | 极高 | 完整模型 | 多卡生产训练 |

推荐选择顺序：

1. 首先运行 LoRA，确认数据、loss、JSON 输出和评估流程；
2. 显存不足时改为 QLoRA；
3. LoRA 收敛稳定后测试 DoRA；
4. 只有数据量、GPU 和训练基础设施充足时才使用 Full SFT。

## 7. 训练配置与运行

### 7.1 安装和环境检查

```bash
bash scripts/setup_conda.sh
conda activate vlm-qwen
nvidia-smi
```

确认项目依赖：

```bash
python3 -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
```

### 7.2 核心训练配置

`configs/train.yaml` 的主要字段：

| 字段 | 说明 |
|---|---|
| `model_name_or_path` | Hugging Face 模型名或本地模型目录 |
| `finetune_mode` | 默认微调模式 |
| `output_dir` | checkpoint 和最终模型目录 |
| `log_dir` | 文件日志和 Trainer 日志目录 |
| `train_dataset` | 训练数据集配置，可以是单个字典或列表 |
| `validation_dataset` | 验证数据集配置 |
| `max_seq_length` | 最大文本/多模态序列长度 |
| `gradient_accumulation_steps` | 梯度累积步数 |
| `learning_rate` | 学习率 |
| `bf16` | 是否使用 BF16 |
| `save_steps` | checkpoint 保存间隔 |
| `eval_steps` | 验证间隔 |

当前 COCO 配置示例：

```yaml
model_name_or_path: Qwen/Qwen3-VL-8B-Instruct
finetune_mode: lora
output_dir: outputs/qwen3-coco-det-cls
log_dir: outputs/logs

train_dataset:
  type: coco
  annotations: /home/yuxiangzhu/volume/VLM/data/coco_train.jsonl
  media_root: /home/yuxiangzhu/volume/animal_det/data/coco/train2017

validation_dataset:
  type: coco
  annotations: /home/yuxiangzhu/volume/VLM/data/coco_val.jsonl
  media_root: /home/yuxiangzhu/volume/animal_det/data/coco/val2017

num_train_epochs: 3
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.0002
max_seq_length: 4096
bf16: true
```

### 7.3 启动训练

推荐使用统一脚本：

```bash
bash scripts/finetune.sh lora configs/train.yaml outputs/qwen3-coco-lora
```

原始 Python 入口同样可用：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m src.training.train \
  --config configs/train.yaml \
  --mode lora \
  --output-dir outputs/qwen3-coco-lora
```

### 7.4 断点续训

```bash
RESUME_FROM_CHECKPOINT=outputs/qwen3-coco-lora/checkpoint-100 \
GPU_ID=0 bash scripts/finetune.sh \
  lora configs/train.yaml outputs/qwen3-coco-lora
```

断点目录必须和当前微调模式匹配，不要用 LoRA checkpoint 恢复 QLoRA、DoRA 或 Full SFT。

### 7.5 日志和 checkpoint

运行日志写入：

```text
outputs/logs/run.log
```

checkpoint 写入当前 `output_dir`：

```text
outputs/qwen3-coco-lora/
├── checkpoint-100/
├── checkpoint-200/
└── adapter_model.safetensors
```

配置：

```yaml
logging_steps: 5
save_steps: 100
eval_steps: 100
save_total_limit: 2
```

`save_total_limit` 可以防止大量 checkpoint 占满磁盘。训练时重点观察 `loss`、`eval_loss`、学习率、每步耗时、GPU 显存以及生成 JSON 的合法率。

### 7.6 当前监督策略

当前 collator 会复制完整 `input_ids` 作为标签，并只把 padding 位置设置为 `-100`。因此 system、user prompt 和 assistant answer 都参与 loss 计算。这可以运行，但不是最理想的 SFT 策略。

生产训练建议进一步增加 assistant-only loss mask，只监督 assistant JSON 输出。这样能够减少模型学习复述提示词，并提高结构化输出稳定性。

## 8. 微调前后指标 HTML 报告

项目提供 `scripts/report.sh`，会使用同一份验证集分别评估基础模型和微调模型，然后生成一份可直接用浏览器打开的 HTML 对比报告。

```bash
bash scripts/report.sh \\
  Qwen/Qwen3-VL-8B-Instruct \\
  outputs/qwen3-coco-lora \\
  data/coco_val.jsonl \\
  /home/yuxiangzhu/volume/animal_det/data/coco/val2017 \\
  outputs/reports/qwen3_lora_comparison.html \\
  image
```

参数顺序：

```text
参数1：微调前模型路径或 Hugging Face 模型名
参数2：微调后模型路径
参数3：验证集 JSONL，默认 data/coco_val.jsonl
参数4：图片/视频根目录，默认 COCO val2017 目录
参数5：HTML 输出路径，默认 outputs/reports/finetune_comparison.html
参数6：输入类型 image 或 video，默认 image
参数7：视频 FPS，默认 2
```

视频模型对比：

```bash
bash scripts/report.sh \\
  Qwen/Qwen3-VL-8B-Instruct \\
  outputs/qwen3-video-lora \\
  data/video_val.jsonl \\
  /data/videos \\
  outputs/reports/video_comparison.html \\
  video \\
  2
```

报告生成过程：

1. 评估微调前模型；
2. 保存基础模型指标 JSON；
3. 评估微调后模型；
4. 保存微调模型指标 JSON；
5. 生成 HTML 对比报告。

默认会生成：

```text
outputs/reports/finetune_comparison.html
outputs/reports/finetune_comparison.baseline.json
outputs/reports/finetune_comparison.finetuned.json
```

HTML 报告包含：

- 微调前模型和微调后模型信息；
- 验证集样本数；
- Detection F1；
- Classification Accuracy；
- 合法 JSON 输出率；
- 指标绝对变化和百分点变化；
- 指标横向条形图；
- 原始评估 JSON；
- 评估条件一致性提示。

也可以只使用已有的两个指标 JSON 生成报告，不重复执行模型推理：

```bash
python3 -m src.evaluation.report \\
  --before outputs/reports/baseline.json \\
  --after outputs/reports/finetuned.json \\
  --before-name "Qwen3-VL base" \\
  --after-name "Qwen3-VL LoRA" \\
  --output outputs/reports/comparison.html
```

单独保存评估指标：

```bash
python3 -m src.evaluation.evaluate \\
  --model outputs/qwen3-coco-lora \\
  --data data/coco_val.jsonl \\
  --media-root /home/yuxiangzhu/volume/animal_det/data/coco/val2017 \\
  --media-type image \\
  --output-json outputs/reports/lora_metrics.json
```

指标定义：

- `detection_f1`：类别匹配且 IoU 大于等于 0.5 的检测 F1；
- `classification_accuracy`：视频级/图片级类别准确率；
- `json_valid_rate`：模型输出能被解析为合法 JSON 的比例；
- `valid_json`：合法 JSON 样本数；
- `samples`：总评估样本数。

报告只能比较在相同验证集、相同类别集合、相同 IoU 阈值、相同媒体类型和相同采样 FPS 下得到的结果。报告生成不会伪造指标；必须先实际运行两个模型的评估。

## 9. 推理和评估

单视频推理：

```bash
python3 -m src.inference.infer \\
  --model outputs/qwen3-video-det-cls \\
  --media videos/example.mp4 \\
  --media-type video \\
  --fps 2 \\
  --instruction '检测视频中的所有目标，并从 [person, car, bicycle, other] 中选择视频级类别。' \\
  --output prediction.json
```

在验证集上计算分类准确率和 IoU=0.5 的检测 F1：

```bash
python3 -m src.evaluation.evaluate \\
  --model outputs/qwen3-video-det-cls \\
  --data data/val.jsonl
```

评估指标只是基线：真正的检测任务建议补充 mAP@50、mAP@50:95、按类别 AP、空目标召回率以及长视频/短视频分桶指标。

## 10. 目录结构

```text
.
├── configs/train.yaml       # 训练超参数
├── data/example.jsonl       # 标注样例
├── scripts/split_dataset.py # JSONL 切分
├── src/common/              # 公共 schema、日志
├── src/data/                # 数据集适配器和数据协议
├── src/model/               # 模型构建扩展点
├── src/training/            # 训练器和数据 collator
├── src/inference/           # 单图片/视频推理
└── src/evaluation/          # IoU、F1 和评估入口
```

## 11. 常见问题

### 模型名不是 Qwen3-VL

将 `model_name_or_path` 改为本地模型目录或 Hugging Face checkpoint。模型必须支持 `AutoProcessor`、视频输入和生成式输出；如果 checkpoint 使用专用 model class，请在 `src/training/train.py` 和 `src/inference/infer.py` 将 `AutoModelForCausalLM` 替换成该 class。

### 显存不足

依次尝试：降低 `video_fps`、缩短视频、减小 `max_seq_length`、增大梯度累积并保持 batch size 为 1；然后再启用 4-bit bitsandbytes QLoRA。不要同时把视频采样帧、分辨率和序列长度设置得很大。

### 输出不是合法 JSON

增加更多严格 JSON 的 assistant 标注，保持 system prompt 与标注格式一致，并使用 `do_sample=False`。上线前应对输出进行 `parse_json`、坐标范围和类别白名单校验，非法结果进入重试或人工审核队列。
