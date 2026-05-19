# da_lab

## 环境配置命令

```bash
conda create -n coop-da python=3.10 -y
conda activate coop-da

git clone https://github.com/Guldfisk5682/da_lab.git
cd da_lab

git clone https://github.com/KaiyangZhou/Dassl.pytorch.git ../Dassl.pytorch

pip install -e ../Dassl.pytorch
pip install -r requirements.txt
```

说明：

- `requirements.txt` 已包含：
  - `torch==2.6.0+cu124`
  - `torchvision==0.21.0+cu124`
  - `--extra-index-url https://download.pytorch.org/whl/cu124`
- 因此远端服务器按上面顺序执行即可。
- `Dassl.pytorch` 不是 `requirements.txt` 里的普通 pip 包，必须单独 clone 并 `pip install -e`。

## 训练命令

### Stage 1

只训练浅层适配模块和 gate：

```bash
export DATA=/path/to/datasets
export SOURCE_DOMAIN=amazon
export TARGET_DOMAIN=webcam
export SEED=1
export STAGE=1

bash scripts/cocoop_da/office31_train.sh
```

### Stage 2

训练浅层适配模块、gate、CoCoOp prompt learner：

```bash
export DATA=/path/to/datasets
export SOURCE_DOMAIN=amazon
export TARGET_DOMAIN=webcam
export SEED=1
export STAGE=2

bash scripts/cocoop_da/office31_train.sh
```

### 可用超参数

脚本直接使用的环境变量：

- `DATA`: 数据集根目录，例如 `/data/datasets`
- `SOURCE_DOMAIN`: 源域，可选 `amazon` / `dslr` / `webcam`
- `TARGET_DOMAIN`: 目标域，可选 `amazon` / `dslr` / `webcam`
- `SEED`: 随机种子
- `STAGE`: `1` 或 `2`
- `OUTPUT_DIR`: 可选，自定义输出目录

如果要覆盖配置文件中的训练超参数，可以直接这样传：

```bash
python train.py \
  --root "${DATA}" \
  --seed "${SEED}" \
  --trainer CoCoOpDAV0 \
  --dataset-config-file configs/datasets/office31.yaml \
  --config-file configs/trainers/CoCoOpDA/vit_b16_v0.yaml \
  --source-domains "${SOURCE_DOMAIN}" \
  --target-domains "${TARGET_DOMAIN}" \
  TRAINER.COCOOP.N_CTX 4 \
  OPTIM.LR 0.002 \
  OPTIM.MAX_EPOCH 10
```

当前主配置文件：

```text
configs/trainers/CoCoOpDA/vit_b16_v0.yaml
```

当前关键默认值：

- `TRAINER.COCOOP.N_CTX = 4`
- `TRAINER.COCOOP.CTX_INIT = "a photo of a"`
- `OPTIM.LR = 0.002`
- `OPTIM.MAX_EPOCH = 10`
- `TRAINER.COCOOP_DA.INJECT_LAYER = 3`
- `TRAINER.COCOOP_DA.ADAPT_MODE = "s2t"`

## 评测命令

```bash
export DATA=/path/to/datasets
export SOURCE_DOMAIN=amazon
export TARGET_DOMAIN=webcam
export SEED=1
export STAGE=1
export MODEL_DIR=output/office31/cocoop_da_v0/A2W/seed1/stage1

bash scripts/cocoop_da/office31_eval.sh
```

### 评测可用超参数

- `DATA`: 数据集根目录
- `SOURCE_DOMAIN`: 源域
- `TARGET_DOMAIN`: 目标域
- `SEED`: 随机种子
- `STAGE`: `1` 或 `2`
- `MODEL_DIR`: 待评测 checkpoint 目录
- `LOAD_EPOCH`: 可选，指定加载某个 epoch；不填则默认读 best/last 逻辑

如果你要直接跑六个 Office-31 任务，只需要替换：

- `SOURCE_DOMAIN`
- `TARGET_DOMAIN`
- `STAGE`
- `SEED`
