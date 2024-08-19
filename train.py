#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as mcolors
from torchvision.transforms import ToPILImage
from torch.nn import functional as F

import cv2
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Set
from detectron2.solver.build import maybe_add_gradient_clipping
import itertools
import copy
import torch.distributed as dist

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler


from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    MetadataCatalog,
)
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    print_csv_format,
    SemSegEvaluator,
)
from detectron2.modeling import build_model
from detectron2.utils.events import EventStorage
from torch.nn.parallel import DistributedDataParallel
from detectron2.utils.logger import setup_logger
from skimage.segmentation import find_boundaries
from skimage.transform import resize
from skimage.color import label2rgb
import numpy as np
from skimage.segmentation import mark_boundaries
import sys

from tgmformer import (
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)
logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i) # 打印格式 copypaste:...
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def build_optimizer(cfg, model):
    weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
    weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

    defaults = {}
    defaults["lr"] = cfg.SOLVER.BASE_LR
    defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if "backbone" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
            if (
                "relative_position_bias_table" in module_param_name
                or "absolute_pos_embed" in module_param_name
            ):
                # print(module_param_name)
                hyperparams["weight_decay"] = 0.0
            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = weight_decay_norm
            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = weight_decay_embed
            params.append({"params": [value], **hyperparams})

    def maybe_add_full_model_gradient_clipping(optim):
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
            and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
    elif optimizer_type == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, cfg.SOLVER.BASE_LR
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer

def build_train_loader(cfg, id):
    print("cfg.DATASETS.TRAIN[id]:", cfg.DATASETS.TRAIN[id])
    # Semantic segmentation dataset mapper
    if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
        mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, dataset_name = cfg.DATASETS.TRAIN[id], mapper=mapper)
    # Panoptic segmentation dataset mapper
    elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
        mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, dataset_name = cfg.DATASETS.TRAIN[id], mapper=mapper)
    # Instance segmentation dataset mapper
    elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
        mapper = MaskFormerInstanceDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, dataset_name = cfg.DATASETS.TRAIN[id], mapper=mapper)
    else:
        mapper = None
        return build_detection_train_loader(cfg, dataset_name = cfg.DATASETS.TRAIN[id], mapper=mapper)


def map_weights_to_full_res(hard_assignments, w):
    B, H, W = hard_assignments.shape

    B, _, superpixel_h, superpixel_w = w.shape
    # Flatten w to make indexing easier
    flat_w = w.view(B, -1)  # [B, superpixel_h * superpixel_w]

    # We assume that hard_assignments have been correctly scaled to index into flat_w
    # Map full resolution pixels to their corresponding superpixel weights
    full_res_weights = flat_w.gather(1, hard_assignments.view(B, -1))   # Flatten hard_assignments for indexing
    full_res_weights = full_res_weights.view(B, H, W)  # Reshape back to the original image dimensions

    return full_res_weights

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def postprocess_activations(activations):
    output = activations
    output *= 255
    return 255 - output.astype('uint8')

def print_model_parameter(model):
    for name, param in model.named_parameters():
            if param.grad is not None and not torch.allclose(param.grad, torch.zeros_like(param.grad)):
                # print(f"Layer name: {name}")
                print(f'Layer: {name}, Gradient of Input: {param.grad}')

def check_grad_status(model):
    for name, param in model.named_parameters():
        if 'discriminator' in name or 'contrastive_proj' in name or 'linear_fuse' in name:
            print(f"{name} requires_grad: {param.requires_grad}")

def monitor_parameters(model, iter_num):
    if iter_num % 10 == 0:
        print("Parameter check at iteration:", iter_num)
        for name, param in model.named_parameters():
            if 'discriminator' in name or 'contrastive_proj' in name or 'linear_fuse' in name:
                print(f"{name} - Mean: {param.data.mean()}, Std Dev: {param.data.std()}")

def do_train_discriminator(cfg, model, max_iter, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    if resume:
        checkpointer = DetectionCheckpointer(
            model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
        )
        start_iter = (
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False).get(
                "iteration", -1
            )
            + 1
        )
        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
        )
    start_iter = 0
    print("resume:", resume)

    writers = (
        default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    )
    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_target_loader = build_train_loader(cfg,0)
    data_source_loader = build_train_loader(cfg,1)
    logger.info("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter) as storage:
        for (data_target, data_source), iteration in zip(zip(data_target_loader, data_source_loader), range(start_iter, max_iter)):
            storage.iter = iteration
            optimizer.zero_grad()

            d_pixel_target, hard_assignments = model(data_target, True)
            dloss_t = torch.mean((1 - d_pixel_target) ** 2)
            variance_t = torch.var(d_pixel_target)

            dloss_t.backward()
            optimizer.step()
            # 训练源数据
            optimizer.zero_grad()
            d_pixel_source, hard_assignments = model(data_source, True)
            dloss_s = torch.mean(d_pixel_source ** 2)
            variance_s = torch.var(d_pixel_source)
            dloss_s.backward()
            optimizer.step()

            # 存lr
            combined_loss = dloss_s.item() + dloss_t.item()
            combined_variance = variance_s.item() + variance_t.item()
            storage.put_scalar("dloss_s", dloss_s.item(), smoothing_hint=False)
            storage.put_scalar("dloss_t", dloss_t.item(), smoothing_hint=False)
            storage.put_scalar("combined_loss", combined_loss, smoothing_hint=False)
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            scheduler.step()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()

            if comm.get_world_size() > 1:
                combined_loss_tensor = torch.tensor(combined_loss).cuda()
                dist.all_reduce(combined_loss_tensor, op=dist.ReduceOp.MIN)
                combined_loss = combined_loss_tensor.item()
 
        for param in model.module.sem_seg_head.predictor.discriminator.parameters():
            param.requires_grad = False
        for param in model.module.sem_seg_head.predictor.contrastive_proj.parameters():
            param.requires_grad = False
        for param in model.module.sem_seg_head.predictor.linear_fuse.parameters():
            param.requires_grad = False




def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER

    writers = (
        default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_target_loader = build_train_loader(cfg,0)
    data_source_loader = build_train_loader(cfg,1)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for (data_target, data_source), iteration in zip(zip(data_target_loader, data_source_loader), range(start_iter, max_iter)):
            storage.iter = iteration

            optimizer.zero_grad()
            loss_dict_target, d_pixel_target, hard_assignments = model(data_target)

            assert torch.isfinite(losses_target).all(), "Losses are not finite"
            losses_target.backward()
            loss_dict_reduced = {
            k: v.item() for k, v in comm.reduce_dict(loss_dict_target).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
            optimizer.step()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False
            )
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()

def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa
    if comm.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    print("return ddp")
    return ddp

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)
    cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
    distributed = comm.get_world_size() > 1
    if distributed:
        model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=True)
    do_train_discriminator(cfg, model, 50, resume=True)
    do_train(cfg, model)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )