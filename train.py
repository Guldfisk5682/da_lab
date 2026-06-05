import argparse
import os
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet
import datasets.office31
import datasets.office_home_mtda

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.cocoop_mtda
import trainers.cocoop_vpt_mtda
import trainers.style_prompt_mtda
import trainers.zsclip


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.TRAINER.COCOOP_MTDA = CN()
    cfg.TRAINER.COCOOP_MTDA.PREC = "fp16"
    cfg.TRAINER.COCOOP_MTDA.DEBUG = CN()
    cfg.TRAINER.COCOOP_MTDA.DEBUG.PRINT_ONCE = False

    cfg.TRAINER.COCOOP_VPT_MTDA = CN()
    cfg.TRAINER.COCOOP_VPT_MTDA.PREC = "fp16"
    cfg.TRAINER.COCOOP_VPT_MTDA.N_VCTX = 4
    cfg.TRAINER.COCOOP_VPT_MTDA.VCTX_INIT_STD = 0.02
    cfg.TRAINER.COCOOP_VPT_MTDA.VISION_PROMPT_DEPTH = 1
    cfg.TRAINER.COCOOP_VPT_MTDA.VCTX_POSITION = "append"
    cfg.TRAINER.COCOOP_VPT_MTDA.DOMAIN_AWARE = CN()
    cfg.TRAINER.COCOOP_VPT_MTDA.DOMAIN_AWARE.ENABLED = False
    cfg.TRAINER.COCOOP_VPT_MTDA.DOMAIN_AWARE.TEXT_TEMPLATE = "a {domain} image."
    cfg.TRAINER.COCOOP_VPT_MTDA.DOMAIN_AWARE.HIDDEN_DIM = 512
    cfg.TRAINER.COCOOP_VPT_MTDA.DOMAIN_AWARE.GAMMA_INIT = 0.0
    cfg.TRAINER.COCOOP_VPT_MTDA.DOMAIN_AWARE.GAMMA_LEARNABLE = True
    cfg.TRAINER.COCOOP_VPT_MTDA.DEBUG = CN()
    cfg.TRAINER.COCOOP_VPT_MTDA.DEBUG.PRINT_ONCE = False

    cfg.TRAINER.STYLE_PROMPT_MTDA = CN()
    cfg.TRAINER.STYLE_PROMPT_MTDA.PREC = "fp16"
    cfg.TRAINER.STYLE_PROMPT_MTDA.DEBUG = CN()
    cfg.TRAINER.STYLE_PROMPT_MTDA.DEBUG.PRINT_ONCE = False

    cfg.TRAINER.STYLE_PROMPT = CN()
    cfg.TRAINER.STYLE_PROMPT.ENABLED = False
    cfg.TRAINER.STYLE_PROMPT.STYLE_LAYER = 3
    cfg.TRAINER.STYLE_PROMPT.TOKEN_SCOPE = "patch"
    cfg.TRAINER.STYLE_PROMPT.STYLE_QUEUE_SIZE = 8
    cfg.TRAINER.STYLE_PROMPT.SELECTION = "domainwise_top1"
    cfg.TRAINER.STYLE_PROMPT.DISTANCE = "cosine"
    cfg.TRAINER.STYLE_PROMPT.STYLE_MLP_HIDDEN = 128
    cfg.TRAINER.STYLE_PROMPT.DOMAIN_STYLE_MLP_HIDDEN = 128
    cfg.TRAINER.STYLE_PROMPT.BETA_INIT = 0.0
    cfg.TRAINER.STYLE_PROMPT.BETA_LEARNABLE = True
    cfg.TRAINER.STYLE_PROMPT.LOSS = "source_ce_only"
    cfg.TRAINER.STYLE_PROMPT.LAMBDA_ENT = 0.01
    cfg.TRAINER.STYLE_PROMPT.EPS = 1e-6

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        # Respect CUDA_VISIBLE_DEVICES by binding the current process
        # to the first visible logical GPU before any model/data placement.
        torch.cuda.set_device(0)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        print(
            "Using CUDA device: logical cuda:0"
            " "
            f"(CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER', '<unset>')}, "
            f"CUDA_VISIBLE_DEVICES={visible}, "
            f"torch.current_device()={torch.cuda.current_device()}, "
            f"torch.cuda.device_count()={torch.cuda.device_count()})"
        )
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    if args.opts and args.opts[0] == "--":
        args.opts = args.opts[1:]
    main(args)
