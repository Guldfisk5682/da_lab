# da_lab

当前活跃分支方向是 `Office-Home` 的 source-available closed-set SS-MTDA。

## 环境准备

```bash
conda create -n coop-da python=3.10 -y
conda activate coop-da

git clone https://github.com/Guldfisk5682/da_lab.git
cd da_lab

git clone https://github.com/KaiyangZhou/Dassl.pytorch.git ../Dassl.pytorch
pip install -e ../Dassl.pytorch
pip install -r requirements.txt
```

如果服务器还没有可用的 `ViT-B/16` CLIP 权重，可以继续沿用原始 CoOp/CoCoOp 的方式，在训练时通过 `MODEL.BACKBONE.NAME=ViT-B/16` 让代码按需加载。这个分支本地不会主动替你下载。

## 当前保留的主入口

- `trainers/coop.py`
- `trainers/cocoop.py`
- `trainers/cocoop_mtda.py`
- `trainers/style_prompt_mtda.py`
- `datasets/office_home_mtda.py`
- `configs/datasets/office_home_mtda.yaml`
- `configs/trainers/CoCoOpMTDA/vit_b16.yaml`
- `configs/trainers/StylePromptMTDA/vit_b16.yaml`
- `scripts/style_prompt_mtda/run_officehome_one.sh`
- `scripts/style_prompt_mtda/run_officehome_all.sh`
- `scripts/style_prompt_mtda/collect_officehome_results.py`

## 归档说明

旧的 `V0 / V1 / legacy-GSPA / Office-31 ablation` 已移动到：

- `archive/v0_v1_ablation/`
- `archive/legacy_gspa/`

迁移细节见：

- `docs/migration_to_style_prompt_mtda.md`

## 轻量检查

不要下载权重或数据集时，可以先做：

```bash
python train.py --help
```

## Office-Home 数据准备

`Dassl` 文档给出的 `Office-Home` 官方页面是：

```text
http://hemanthdv.org/OfficeHome-Dataset/
```

本仓库现在提供：

- `scripts/datasets/download_officehome.sh`
- `scripts/datasets/verify_officehome_layout.sh`

如果你已经手上有官方压缩包：

```bash
export DATA_ROOT=/workspace/txc/da_lab/data
export OFFICEHOME_ARCHIVE=/path/to/OfficeHomeDataset_10072016.zip
bash scripts/datasets/download_officehome.sh
bash scripts/datasets/verify_officehome_layout.sh
```

脚本默认数据根目录现在就是：

```text
/workspace/txc/da_lab/data
```

如果你什么都不传，脚本会按这个顺序尝试：

1. Office-Home 官方页面中的 Google Drive 下载入口
2. Hugging Face 备选源 `flwrlabs/office-home`

HF fallback 默认使用：

```text
HF_ENDPOINT=https://hf-mirror.com
HF_DATASET_REPO=flwrlabs/office-home
HF_DATASET_SPLIT=train
```

之所以选 `flwrlabs/office-home`，是因为它的数据卡明确给出了：

- `Formats: parquet`
- 共有 `image / domain / label` 三列
- `train` split 共 `15.6k` 行

这正好适合我们在下载脚本里重建为：

```text
office_home/
├── art/
├── clipart/
├── product/
└── real_world/
```

如果你想手工覆盖下载 URL：

```bash
export DATA_ROOT=/workspace/txc/da_lab/data
export OFFICEHOME_URL="https://.../OfficeHomeDataset_10072016.zip"
bash scripts/datasets/download_officehome.sh
bash scripts/datasets/verify_officehome_layout.sh
```

如果你想强制直接走 HF mirror：

```bash
export DATA_ROOT=/workspace/txc/da_lab/data
export OFFICEHOME_SOURCE=hf
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASET_REPO=flwrlabs/office-home
bash scripts/datasets/download_officehome.sh
bash scripts/datasets/verify_officehome_layout.sh
```

准备完成后，目录应为：

```text
DATA_ROOT/
└── office_home/
    ├── art/
    ├── clipart/
    ├── product/
    └── real_world/
```

脚本会自动把官方常见原始结构里的 `Art/Clipart/Product/Real World/images/...` 整理成上面的 `office_home/...` 结构。

## Office-Home MTDA 训练入口

单个 source：

```bash
bash scripts/style_prompt_mtda/run_officehome_one.sh A 1 cocoop_mt
bash scripts/style_prompt_mtda/run_officehome_one.sh A 1 style_prompt
```

最小 smoke test：

```bash
bash scripts/style_prompt_mtda/run_officehome_one.sh A 1 cocoop_mt --debug
bash scripts/style_prompt_mtda/run_officehome_one.sh A 1 style_prompt --debug
```

四个 source 全部运行：

```bash
bash scripts/style_prompt_mtda/run_officehome_all.sh cocoop_mt
bash scripts/style_prompt_mtda/run_officehome_all.sh style_prompt
```

结果汇总：

```bash
python scripts/style_prompt_mtda/collect_officehome_results.py
```

`OfficeHomeMTDA` 不会物理重写数据集文件，而是在运行时自动组织成：

- `1` 个 labeled source loader
- `3` 个 unlabeled target loaders
- `3` 个 target test loaders

也就是说，SS-MTDA protocol 是由 dataset wrapper 和 trainer 在运行时完成的，不需要你手工重新打包数据。
