# DomainNet baseline continuation on a 10 GB GPU

The remaining reproduction runs are CoOp and CoCoOp source-only for all six
DomainNet source domains. CLIP zero-shot and original MaPLe source-only are
already complete and do not need to be repeated.

## Data

Use one common parent directory, for example `/data/datasets`:

```bash
export DATA_ROOT=/data/datasets
bash scripts/datasets/download_domainnet.sh
python scripts/datasets/verify_domainnet_layout.py --root "$DATA_ROOT"
```

The official cleaned archives are roughly 18 GB before extraction. If direct
download is slow, copy the complete `DomainNet/` directory from the old server,
including `DomainNet/image_list/`, and run the same verifier. Do not mix list
files from another DomainNet release.

The first CLIP run may download ViT-B/16 weights. On a network-restricted host,
copy the existing CLIP cache from the old server before launching.

## Low-memory run

Activate the same environment used by this repository, then run:

```bash
export DATA_ROOT=/data/datasets
export CUDA_VISIBLE_DEVICES=0
bash scripts/prompt_baselines_mtda/run_domainnet_10gb_queue.sh \
  2>&1 | tee domainnet_10gb_queue.log
```

Defaults are CoOp train batch 16, CoCoOp train batch 2, test batch 16, and four
workers. They override batch sizes only at launch; original trainer configs and
training logic remain unchanged. If the 10 GB card still runs out of memory,
restart with:

```bash
COOP_TRAIN_BATCH_SIZE=8 \
COCOOP_TRAIN_BATCH_SIZE=1 \
TEST_BATCH_SIZE=8 \
bash scripts/prompt_baselines_mtda/run_domainnet_10gb_queue.sh
```

Each completed task writes `mtda_metrics.json`; rerunning the queue skips such
tasks and continues from the first unfinished source. A failed task stops the
serial queue, so a downstream run cannot start after an OOM or traceback.
