# Office-31 Setup

This document captures the Phase 2 dataset work without downloading the dataset during this rollout.

## Supported directory layouts

The local dataset wrapper supports either of these layouts under `DATASET.ROOT`:

```text
$DATA/office31/amazon/<class_name>/*.jpg
$DATA/office31/dslr/<class_name>/*.jpg
$DATA/office31/webcam/<class_name>/*.jpg
```

or:

```text
$DATA/office31/amazon/images/<class_name>/*.jpg
$DATA/office31/dslr/images/<class_name>/*.jpg
$DATA/office31/webcam/images/<class_name>/*.jpg
```

## Domain configuration

The dataset wrapper uses standard `Dassl` domain fields:

```bash
--source-domains amazon
--target-domains webcam
```

Example tasks:

- `amazon -> webcam`
- `amazon -> dslr`
- `webcam -> amazon`
- `webcam -> dslr`
- `dslr -> amazon`
- `dslr -> webcam`

## Data semantics

- `train_x`: labeled source-domain images
- `train_u`: unlabeled target-domain images
- `test`: target-domain evaluation images

Target labels are present in the dataset object for evaluation bookkeeping, but the DA trainer does not consume them for supervised loss.

## Relevant files

- `datasets/office31.py`
- `configs/datasets/office31.yaml`
- `scripts/datasets/download_office31.sh`
- `scripts/cocoop_da/office31_train.sh`
- `scripts/cocoop_da/office31_eval.sh`

## Notes

- `scripts/datasets/download_office31.sh` is intentionally an extraction entrypoint rather than an auto-downloader, so we do not hard-code an unverified archive URL.
- Real dataset download and extraction should happen on the remote server before training.
