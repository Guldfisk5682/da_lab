# Codebase Notes

These notes cover the Phase 1 inspection work requested in `AGENT.md`.

## 1. How CoCoOp builds the CLIP model

- `train.py` imports `trainers.cocoop`, and `Dassl` builds the trainer from `cfg.TRAINER.NAME`.
- `trainers/cocoop.py` uses `load_clip_to_cpu(cfg)` to resolve the CLIP backbone name from `cfg.MODEL.BACKBONE.NAME`.
- `load_clip_to_cpu(cfg)` calls `clip._download(url)` and then `clip.build_model(...)`.
- The resulting CLIP object is wrapped by `CustomCLIP`.

## 2. Where the visual encoder is called

- In `trainers/cocoop.py`, `CustomCLIP.forward()` calls `self.image_encoder(image.type(self.dtype))`.
- `self.image_encoder` is `clip_model.visual`.
- For ViT backbones, the implementation lives in `clip/model.py` under `VisionTransformer`.
- The ViT path is:
  1. patch embedding by `visual.conv1`
  2. class token + positional embedding
  3. `visual.ln_pre`
  4. transformer blocks in `visual.transformer.resblocks`
  5. `visual.ln_post`
  6. optional projection `visual.proj`

## 3. Where image features are passed into the prompt learner

- `CustomCLIP.forward()` normalizes image features first.
- The normalized image feature is passed to `self.prompt_learner(image_features)`.
- Inside `PromptLearner.forward()`, `meta_net` maps the image feature to an instance-conditioned bias for context tokens.
- This is the exact point where CoCoOp ties image features to prompt construction.

## 4. Where trainable parameters are selected

- In `CoCoOp.build_model()`, all parameters are frozen except names containing `prompt_learner`.
- The optimizer is built only on `self.model.prompt_learner`.
- This makes the official CoCoOp baseline prompt-only fine-tuning by default.
- The new `CoCoOpDAV0` trainer follows the same pattern but selects:
  - Stage 1: `shallow_adapt` and `gate`
  - Stage 2: `shallow_adapt`, `gate`, and `prompt_learner`

## 5. How datasets are registered

- `train.py` explicitly imports each dataset module so the `DATASET_REGISTRY` side effect runs before `Dassl` builds the dataset.
- `Dassl` resolves datasets via `build_dataset(cfg)`.
- Domain adaptation datasets use `cfg.DATASET.SOURCE_DOMAINS` and `cfg.DATASET.TARGET_DOMAINS`.
- `DatasetBase` supports:
  - `train_x`: labeled source data
  - `train_u`: unlabeled target data
  - `test`: evaluation split

## 6. ViT tensor layout relevant to V0

- `VisionTransformer` receives tokens in `[B, L, C]` before entering the transformer.
- Internally, tokens are permuted to `[L, B, C]` before `transformer.resblocks`.
- After the transformer, they are permuted back to `[B, L, C]`.
- This matches the V0 requirement to inject after block 3 and then continue through blocks 4..12.

## 7. Office-31 implication

- `CoOp` itself does not ship an Office-31 dataset module.
- `Dassl.pytorch` does have an Office-31 DA dataset, so this repo now mirrors that registration pattern locally through `datasets/office31.py`.
- The local wrapper keeps the project self-contained and makes the train entrypoint explicit.
