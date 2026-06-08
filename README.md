# da_lab

当前活跃方向是 `Office-Home` 的 source-available closed-set SS-MTDA。

当前主线已切换到 `CLIPVPTMTDA`：先脱离 CoCoOp，评测 frozen CLIP zero-shot 和 CLIP + persistent VCTX 在 SS-MTDA 上的表现。`CoCoOpVPTMTDA` 保留为上一阶段强基线与负结果参考。

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
- `trainers/cocoop_vpt_mtda.py`
- `trainers/clip_tssp_mtda.py`
- `trainers/clip_vpt_mtda.py`
- `trainers/style_prompt_mtda.py`
- `models/clip_tssp.py`
- `models/clip_vpt.py`
- `models/visual_prompt.py`
- `datasets/office_home_mtda.py`
- `configs/datasets/office_home_mtda.yaml`
- `configs/trainers/CoCoOpMTDA/vit_b16.yaml`
- `configs/trainers/CoCoOpVPTMTDA/vit_b16.yaml`
- `configs/trainers/CLIPTSSPMTDA/vit_b16.yaml`
- `configs/trainers/CLIPVPTMTDA/vit_b16.yaml`
- `configs/trainers/StylePromptMTDA/vit_b16.yaml`
- `scripts/clip_tssp_mtda/run_officehome_one.sh`
- `scripts/clip_tssp_mtda/run_officehome_all.sh`
- `scripts/clip_tssp_mtda/collect_officehome_results.py`
- `scripts/clip_vpt_mtda/run_officehome_one.sh`
- `scripts/clip_vpt_mtda/run_officehome_all.sh`
- `scripts/clip_vpt_mtda/collect_officehome_results.py`
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

当前新阶段优先跑 CLIP-first B0/B1。

B0: frozen CLIP zero-shot，仅评测：

```bash
bash scripts/clip_vpt_mtda/run_officehome_all.sh clip_zs
```

B1: CLIP + persistent VCTX，只训练视觉 context tokens：

```bash
bash scripts/clip_vpt_mtda/run_officehome_all.sh clip_vpt
```

单个 source smoke test：

```bash
bash scripts/clip_vpt_mtda/run_officehome_one.sh A 1 clip_zs --debug
bash scripts/clip_vpt_mtda/run_officehome_one.sh A 1 clip_vpt --debug
```

CLIP-first 结果汇总：

```bash
python scripts/clip_vpt_mtda/collect_officehome_results.py
```

基础版 `CLIPTSSPMTDA` 不使用 VCTX，而是把 frozen CLIP ViT 的 12 层 hidden-state mean/std 映射成 text-side style tokens。后续 vision-side 对照可显式启用 persistent VCTX。

full: source-style tokens + target-set style tokens + target-source gap tokens：

```bash
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_full
```

no-gap: source-style tokens + target-set style tokens：

```bash
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_no_gap
```

新增三组 prompt-order / layer-compression 消融：

```bash
# 12-layer tokens: [source, gap, target]
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_gap

# Pair adjacent layers into 6 tokens: [source, target]
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_pair

# Pair adjacent layers into 6 tokens: [source, gap, target]
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_pair_gap
```

进一步压缩实验均保留 middle-gap `[source, gap, target]`：

```bash
# 只压缩 source/target style tokens，gap 保持 12 层
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_style3_gap1
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_style4_gap1

# source/target style 与 gap 同尺度压缩
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_style3_gap3
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_style4_gap4

# source/target style 保持 12 层，只压缩 gap
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_style1_gap3
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_style1_gap4
```

`STYLE_GROUP_SIZE=3/4` 表示每三/四个相邻 ViT layer style tokens 固定取均值；`GAP_GROUP_SIZE` 独立控制 gap token 的分组尺度。

固定当前最佳 PairGap 后，加入当前 source/test image 自身的多层 content tokens：

```bash
# [S6, G6, T6, I12]
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_pair_gap_img12

# [S6, G6, T6, I6]，相邻两层 image tokens 取均值
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_pair_gap_img6

# [S6, G6, T6, I4]，相邻三层 image tokens 取均值
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_pair_gap_img4
```

每层 image token 使用完整 hidden tokens 的均值池化，再经过该层独立的 `Linear(vision_dim, text_dim)`。AD-CLIP 源码中的四个 image tokens 并不是相邻三层分组；本仓库使用明确的相邻层分组，以便比较 12/6/4 层内容保留尺度。

固定 `lambda_em=0.01` 的 target conditional entropy 对照：

```bash
# PairGap + L_em，不使用 image tokens
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_pair_gap_em

# PairGap + Img6 + L_em
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_pair_gap_img6_em
```

`L_em` 对三个目标域分别计算标准样本条件熵，再对域等权平均。CLIP 主体保持冻结；该系数不是可学习参数，`LAMBDA_EM=0.0` 时完全跳过目标 logits 路径。

PairGap 与 persistent VCTX8 的 vision-side tuning 对照：

```bash
# C0: source CE 同时更新 PairGap projector 与 persistent VCTX8
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_pair_gap_vctx8

# C1: 增加固定 0.01 target entropy；target text features detach
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_pair_gap_vctx8_em_detach
```

两条路径共用 frozen CLIP。clean visual forward 只为 PairGap 提取 hidden-state style；VCTX forward 生成用于分类的最终 image feature。C1 的 `L_em` 不更新 style projector，只沿 target final image feature 更新共享 VCTX。

PairGap + AdamW 优化器对照：

```bash
# O0: PairGap + AdamW lr=2e-3
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_pair_gap_adamw2e3

# O1: PairGap + AdamW lr=1e-4
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_pair_gap_adamw1e4
```

AdamW 变体使用独立输出目录，避免覆盖或加载已有 `clip_tssp_pair_gap` 的 SGD checkpoint。
当前 optimizer 默认带 1 epoch constant warmup：`WARMUP_EPOCH=1`、`WARMUP_CONS_LR=1e-5`，之后使用 cosine scheduler。

PairGap + frozen CLIP 约束对照：

```bash
# K0: PairGap + AdamW lr=1e-4 + KL(student || frozen CLIP reference)
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_pair_gap_kl

# P0: PairGap + AdamW lr=1e-4 + conservative frozen-CLIP pseudo labels
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_pair_gap_pl

# KP0: PairGap + AdamW lr=1e-4 + KL + pseudo labels
bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_pair_gap_pl_kl
```

KL/PL 变体不额外加载第二份 CLIP；reference logits 使用 frozen clean image feature 与 frozen zero-shot text prompt 计算。PL 只在 frozen CLIP 高置信且当前 student 低置信的 target 样本上生效，默认 `PL_THRESHOLD=0.7`、`PL_STUDENT_THRESHOLD=0.7`。

一晚连续跑五个 PairGap 变体：

```bash
# 默认 seed1，依次跑 O0/O1/K0/P0/KP0，结束后自动 collect 和画 TensorBoard 曲线
bash scripts/clip_tssp_mtda/run_pairgap_5variants.sh

# 如需换 seed
SEED=2 bash scripts/clip_tssp_mtda/run_pairgap_5variants.sh
```

如需诊断 target accuracy 是否随 epoch 震荡，可以开启逐 epoch eval。该开关只用于曲线诊断，正常汇报结果建议保持默认 final-test-only：

```bash
EVAL_EVERY_EPOCH=1 bash scripts/clip_tssp_mtda/run_officehome_all.sh clip_tssp_pair_gap_adamw1e4
EVAL_EVERY_EPOCH=1 bash scripts/clip_tssp_mtda/run_pairgap_5variants.sh
```

单个 source smoke test：

```bash
bash scripts/clip_tssp_mtda/run_officehome_one.sh A 1 clip_tssp_full --debug
bash scripts/clip_tssp_mtda/run_officehome_one.sh A 1 clip_tssp_no_gap --debug
```

TSSP 结果汇总：

```bash
python scripts/clip_tssp_mtda/collect_officehome_results.py
```

TensorBoard 曲线汇总图：

```bash
# 单个方法，输出 results/officehome_tensorboard_clip_tssp_pair_gap_seeds1_eval.png 与 train.png
python scripts/clip_tssp_mtda/plot_tensorboard_curves.py --method clip_tssp_pair_gap --seed 1

# 多个方法同图比较
python scripts/clip_tssp_mtda/plot_tensorboard_curves.py \
  --methods clip_tssp_pair_gap clip_tssp_pair_gap_vctx8 clip_tssp_pair_gap_em \
  --seed 1

# 强制从 log.txt 解析 eval 曲线；默认 auto 会选择点更多的数据源
python scripts/clip_tssp_mtda/plot_tensorboard_curves.py \
  --method clip_tssp_pair_gap_adamw1e4 \
  --seed 1 \
  --eval-source log
```

绘图脚本会覆盖同名旧图片，并同时写出对应的 `_summary.csv`。评测图展示每个 source task 下各 target domain 的 accuracy/macro 曲线；训练图展示 loss、lr、source acc、style/gap norm、pseudo-label/KL/coverage 等重要标量。

collector 会输出逐 source 结果、按目标域 `A/C/P/R` 聚合的平均准确率，以及全部 12 个迁移任务的 overall average。

上一阶段 CoCoOp/StylePrompt/VPT 入口如下，主要用于对照历史结果。

单个 source：

```bash
bash scripts/style_prompt_mtda/run_officehome_one.sh A 1 cocoop_mt
bash scripts/style_prompt_mtda/run_officehome_one.sh A 1 cocoop_vpt
bash scripts/style_prompt_mtda/run_officehome_one.sh A 1 style_prompt
```

最小 smoke test：

```bash
bash scripts/style_prompt_mtda/run_officehome_one.sh A 1 cocoop_mt --debug
bash scripts/style_prompt_mtda/run_officehome_one.sh A 1 cocoop_vpt --debug
bash scripts/style_prompt_mtda/run_officehome_one.sh A 1 style_prompt --debug
```

四个 source 全部运行：

```bash
bash scripts/style_prompt_mtda/run_officehome_all.sh cocoop_mt
bash scripts/style_prompt_mtda/run_officehome_all.sh cocoop_vpt
bash scripts/style_prompt_mtda/run_officehome_all.sh style_prompt
```

VPT 超参扫描建议用 `METHOD_TAG` 分开输出目录，避免覆盖或 resume 到旧实验：

```bash
METHOD_TAG=cocoop_vpt_ctx4_d1 \
EXTRA_OPTS="${EXTRA_OPTS} TRAINER.COCOOP_VPT_MTDA.N_VCTX 4 TRAINER.COCOOP_VPT_MTDA.VISION_PROMPT_DEPTH 1" \
bash scripts/style_prompt_mtda/run_officehome_all.sh cocoop_vpt

METHOD_TAG=cocoop_vpt_ctx4_d3 \
EXTRA_OPTS="${EXTRA_OPTS} TRAINER.COCOOP_VPT_MTDA.N_VCTX 4 TRAINER.COCOOP_VPT_MTDA.VISION_PROMPT_DEPTH 3" \
bash scripts/style_prompt_mtda/run_officehome_all.sh cocoop_vpt

METHOD_TAG=cocoop_vpt_ctx8_d1 \
EXTRA_OPTS="${EXTRA_OPTS} TRAINER.COCOOP_VPT_MTDA.N_VCTX 8 TRAINER.COCOOP_VPT_MTDA.VISION_PROMPT_DEPTH 1" \
bash scripts/style_prompt_mtda/run_officehome_all.sh cocoop_vpt
```

Instance-aware PVC 使用进入 ViT block 前的早期 patch tokens 为每张图生成视觉 prompt tokens，`BETA_INIT=0.0`，训练初始严格等价于当前 persistent VCTX。

`residual` 模式把 instance tokens 加到 shared VCTX 上：

```bash
METHOD_TAG=cocoop_vpt_ctx8_instance_residual \
EXTRA_OPTS="TRAINER.COCOOP_VPT_MTDA.N_VCTX 8 TRAINER.COCOOP_VPT_MTDA.VISION_PROMPT_DEPTH 1 TRAINER.COCOOP_VPT_MTDA.INSTANCE_AWARE.ENABLED True TRAINER.COCOOP_VPT_MTDA.INSTANCE_AWARE.MODE residual" \
bash scripts/style_prompt_mtda/run_officehome_all.sh cocoop_vpt
```

`append` 模式把 instance tokens 追加在 shared VCTX 后面，更接近 ViaPT 的 prompt token 分工：

```bash
METHOD_TAG=cocoop_vpt_ctx8_instance_append \
EXTRA_OPTS="TRAINER.COCOOP_VPT_MTDA.N_VCTX 8 TRAINER.COCOOP_VPT_MTDA.VISION_PROMPT_DEPTH 1 TRAINER.COCOOP_VPT_MTDA.INSTANCE_AWARE.ENABLED True TRAINER.COCOOP_VPT_MTDA.INSTANCE_AWARE.MODE append" \
bash scripts/style_prompt_mtda/run_officehome_all.sh cocoop_vpt
```

该模块的生成路径是：

```text
early patch tokens E0
-> Conv1x1 + GELU + Conv3x3 + GELU + AvgPool
-> Linear predicts mean/log_std
-> reparameterized instance tokens
-> residual: shared VCTX + beta * instance tokens
-> append: [shared VCTX, beta * instance tokens]
```

如果只想验证 target objective 是否能推动当前最好的 VCTX8，不新增任何 prompt 结构，可以开启 target information maximization：

```bash
METHOD_TAG=cocoop_vpt_ctx8_target_im \
EXTRA_OPTS="TRAINER.COCOOP_VPT_MTDA.N_VCTX 8 TRAINER.COCOOP_VPT_MTDA.VISION_PROMPT_DEPTH 1 TRAINER.COCOOP_VPT_MTDA.TARGET_IM.ENABLED True TRAINER.COCOOP_VPT_MTDA.TARGET_IM.LAMBDA_ENT 0.1 TRAINER.COCOOP_VPT_MTDA.TARGET_IM.LAMBDA_DIV 0.1" \
bash scripts/style_prompt_mtda/run_officehome_all.sh cocoop_vpt
```

该 loss 仍然不使用 target labels：

```text
loss = source CE
     + lambda_ent * target sample entropy
     + lambda_div * target class-balance KL
```

结果汇总：

```bash
python scripts/style_prompt_mtda/collect_officehome_results.py
```

如果要一次性补跑 `seed=2/3` 的 `CoCoOpMTDA` 和当前主方法
`cocoop_vpt_ctx8_d1`：

```bash
bash scripts/style_prompt_mtda/run_seed23_cocoop_vpt.sh
```

多 seed 汇总：

```bash
python scripts/style_prompt_mtda/collect_officehome_results.py --seeds 1 2 3
```

`OfficeHomeMTDA` 不会物理重写数据集文件，而是在运行时自动组织成：

- `1` 个 labeled source loader
- `3` 个 unlabeled target loaders
- `3` 个 target test loaders

也就是说，SS-MTDA protocol 是由 dataset wrapper 和 trainer 在运行时完成的，不需要你手工重新打包数据。
