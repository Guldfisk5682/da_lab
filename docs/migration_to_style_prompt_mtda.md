# Migration To Office-Home MTDA And VPT

## 废弃并归档的旧路线

以下旧实验路径已从主目录移出，不再作为当前研究方向的一部分：

### `archive/v0_v1_ablation/`

- `trainers/cocoop_da_v0.py`
- `trainers/cocoop_da_v1.py`
- `models/shallow_adapt.py`
- `configs/trainers/CoCoOpDA/`
- `scripts/cocoop_da/`
- `scripts/parse_office31_results.py`
- `docs/env_setup.md`
- `docs/codebase_notes.md`

这些文件对应的思路包括：

- V0 shallow hidden-state early fusion
- V1 final feature gate
- force_alpha / alpha0 / alpha1
- gate / last3 tuning ablation
- 旧的 Office-31 SS-STDA 脚本与汇总逻辑

### `archive/legacy_gspa/`

- `trainers/gspa_legacy.py`
- `configs/trainers/GSPA_LEGACY/`
- `scripts/gspa_legacy/`
- `scripts/gspa_legacy_ablation/`
- `docs/office31_setup.md`

这些文件对应的思路包括：

- legacy-GSPA hidden-state style swap
- Office-31 ablation study
- style-swap / gate / fixed gate / normal-only 等旧方向

## 当前保留的核心文件

这些文件仍然保留在主目录中，作为新方向的基础设施：

- `trainers/coop.py`
- `trainers/cocoop.py`
- `trainers/checkpoint_utils.py`
- `datasets/office31.py`
- `datasets/office_home_mtda.py`
- `models/clip_vit.py`
- `models/style_prompt.py`
- `scripts/datasets/`
- `scripts/setup/install_dassl.sh`

说明：

- `CoOp / CoCoOp` 主体保持可用；
- 已验证过的数据集注册机制保留；
- Office-31 相关下载脚本保留为历史基础设施，但不再是当前默认研究入口；
- 新方向需要复用的 CLIP ViT shallow-token 工具已从旧实验中抽出，放入 `models/clip_vit.py`；
- 新方向的 style queue / style MLP 工具位于 `models/style_prompt.py`。

## 新方向入口

### 当前活跃主线

`StylePromptMTDA` 已保留为历史实验，但不再作为当前主线继续扩展。
当前活跃方法是：

- `trainers/cocoop_vpt_mtda.py`
- `models/visual_prompt.py`
- `configs/trainers/CoCoOpVPTMTDA/vit_b16.yaml`

当前先验证 independent visual prompt tuning：

```text
CoCoOp text prompt learner + learnable visual ctx injected into CLIP ViT blocks
```

目标是先回答视觉侧 prompt 是否能稳定提升 `CoCoOpMTDA`，再决定是否推进
MaPLe-style text/vision prompt coupling。

`VISION_PROMPT_DEPTH=1` 对应 shallow VPT。更大的 depth 会使用每层独立的
`vctx[layer]`，并在对应 ViT block 前替换 prompt tokens，而不是不断追加 tokens。

### Dataset

- `datasets/office_home_mtda.py`
- `configs/datasets/office_home_mtda.yaml`

### Trainers

- `trainers/cocoop_mtda.py`
- `trainers/cocoop_vpt_mtda.py`
- `trainers/style_prompt_mtda.py`
- `trainers/mtda_base.py`

### Configs

- `configs/trainers/CoCoOpMTDA/vit_b16.yaml`
- `configs/trainers/CoCoOpVPTMTDA/vit_b16.yaml`
- `configs/trainers/StylePromptMTDA/vit_b16.yaml`

### Scripts

- `scripts/datasets/download_officehome.sh`
- `scripts/datasets/verify_officehome_layout.sh`
- `scripts/style_prompt_mtda/run_officehome_one.sh`
- `scripts/style_prompt_mtda/run_officehome_all.sh`
- `scripts/style_prompt_mtda/collect_officehome_results.py`

## Smoke Test

轻量检查：

```bash
python train.py --help
```

最小调试入口：

```bash
bash scripts/style_prompt_mtda/run_officehome_one.sh A 1 cocoop_mt --debug
bash scripts/style_prompt_mtda/run_officehome_one.sh A 1 cocoop_vpt --debug
bash scripts/style_prompt_mtda/run_officehome_one.sh A 1 style_prompt --debug
```

说明：

- `--debug` 模式只跑最小 epoch / 少量 iter；
- 不会自动下载数据集；
- 不会自动下载 CLIP 权重；
- 如果 `DATA` 或 `BACKBONE` 路径不存在，应直接报清晰错误，由远程服务器按需准备。
