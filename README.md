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

## 6. 训练

编辑 `configs/train.yaml`，重点关注模型路径、视频采样帧率、序列长度和 LoRA 参数：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.training.train --config configs/train.yaml
```

训练输出在 `outputs/qwen3-video-det-cls/`。断点续训：

```bash
RESUME_FROM_CHECKPOINT=outputs/qwen3-video-det-cls/checkpoint-100 \\
CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/train.yaml
```

这份实现使用全序列标签监督，padding 部分会被设为 `-100`。生产训练建议再加入 prompt token mask，只计算 assistant 输出损失；如果训练出现模型复述指令或输出格式不稳定，优先做这项改造。

## 7. 推理和评估

单视频推理：

```bash
python -m src.inference.infer \\
  --model outputs/qwen3-video-det-cls \\
  --video videos/example.mp4 \\
  --instruction '检测视频中的所有目标，并从 [person, car, bicycle, other] 中选择视频级类别。' \\
  --output prediction.json
```

在验证集上计算分类准确率和 IoU=0.5 的检测 F1：

```bash
python -m src.evaluation.evaluate \\
  --model outputs/qwen3-video-det-cls \\
  --data data/val.jsonl
```

评估指标只是基线：真正的检测任务建议补充 mAP@50、mAP@50:95、按类别 AP、空目标召回率以及长视频/短视频分桶指标。

## 8. 目录结构

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

## 9. 常见问题

### 模型名不是 Qwen3-VL

将 `model_name_or_path` 改为本地模型目录或 Hugging Face checkpoint。模型必须支持 `AutoProcessor`、视频输入和生成式输出；如果 checkpoint 使用专用 model class，请在 `src/training/train.py` 和 `src/inference/infer.py` 将 `AutoModelForCausalLM` 替换成该 class。

### 显存不足

依次尝试：降低 `video_fps`、缩短视频、减小 `max_seq_length`、增大梯度累积并保持 batch size 为 1；然后再启用 4-bit bitsandbytes QLoRA。不要同时把视频采样帧、分辨率和序列长度设置得很大。

### 输出不是合法 JSON

增加更多严格 JSON 的 assistant 标注，保持 system prompt 与标注格式一致，并使用 `do_sample=False`。上线前应对输出进行 `parse_json`、坐标范围和类别白名单校验，非法结果进入重试或人工审核队列。
