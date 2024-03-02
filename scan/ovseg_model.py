# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Modified by Feng Liang from
# https://github.com/MendelXu/zsseg.baseline/blob/master/mask_former/zero_shot_mask_former_model.py

import logging
from typing import Tuple

import numpy as np
import copy
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from .modeling.clip_adapter import (
    ClipAdapter,
    MaskFormerClipAdapter,
    build_text_prompt,
)
from .maskformer_model import Mask2Former
from .utils.misc import get_gt_binary_masks
from .frequency import LFM, MLP, CA

@META_ARCH_REGISTRY.register()
class SCAN(Mask2Former):
    """
    Main class for zero shot mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        clip_adapter: nn.Module,
        lightweight_transformer: nn.Module,
        LFM: nn.Module,
        fusion_decoder: nn.Module,
        oriclip_mlp: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        clip_ensemble: bool,
        clip_ensemble_weight: float,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        test_topk_per_image: int,
        select_ori_clip_id: list,
        frequency_sigma: list,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            clip_adapter: adapter for clip-based mask classification
            num_queries: int, number of queries
            panoptic_on: bool, whether to output panoptic segmentation prediction
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            semantic_on=semantic_on,
            instance_on=instance_on,
            panoptic_on=panoptic_on,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            test_topk_per_image=test_topk_per_image
        )
        self.clip_adapter: ClipAdapter = clip_adapter

        self.clip_ensemble: bool = clip_ensemble
        self.clip_ensemble_weight: float = clip_ensemble_weight

        self.learnable_weight = nn.Parameter(torch.tensor([0.1]))
        self.lightweight_transformer = lightweight_transformer
        self.LFM = LFM
        self.fusion_decoder = fusion_decoder
        self.oriclip_mlp = oriclip_mlp
        
        self.select_ori_clip_id = select_ori_clip_id
        self.frequency_sigma = frequency_sigma

        self.openai_pixel_mean = torch.Tensor((0.48145466, 0.4578275, 0.40821073)).reshape(3, 1, 1) * 255
        self.openai_pixel_std = torch.Tensor((0.26862954, 0.26130258, 0.27577711)).reshape(3, 1, 1) * 255

    @classmethod
    def from_config(cls, cfg):
        init_kwargs = Mask2Former.from_config(cfg)
        text_templates = build_text_prompt(cfg.MODEL.CLIP_ADAPTER)

        clip_adapter = MaskFormerClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            text_templates,
            mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
            mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
            mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
            mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
            region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
            replace_ratio=cfg.MODEL.CLIP_ADAPTER.REPLACE_RATIO,
            replace_layer=cfg.MODEL.CLIP_ADAPTER.REPLACE_LAYER
        )
        init_kwargs["clip_adapter"] = clip_adapter
        init_kwargs["clip_ensemble"] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE
        init_kwargs[
            "clip_ensemble_weight"
        ] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT

        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=5)
        init_kwargs['lightweight_transformer'] = transformer_encoder
        init_kwargs['LFM'] = LFM(num_channels=1024)
        
        init_kwargs['fusion_decoder'] = CA(input_dim=768, num=5)
        init_kwargs['oriclip_mlp'] = MLP(1024, 768)

        init_kwargs['select_ori_clip_id'] = cfg.MODEL.SELECT_ORI_CLIP_ID
        init_kwargs['frequency_sigma'] = cfg.MODEL.FREQUENCY_SIGMA

        return init_kwargs

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        dataset_name = [x["meta"]["dataset_name"] for x in batched_inputs]
        assert len(set(dataset_name)) == 1
        dataset_name = dataset_name[0]

        images = [x["image"].to(self.device) for x in batched_inputs]

        original_images = [(x - self.openai_pixel_mean.to(images[0].device)) / self.openai_pixel_std.to(images[0].device) for x in images]
        original_images = ImageList.from_tensors(original_images, self.size_divisibility)

        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        # use original clip
        clip_input_img = F.interpolate(original_images.tensor, [224, 224])
        clip_features_all = self.clip_adapter.original_clip(clip_input_img, get_embedding=True)
        select_ori_clip_id = self.select_ori_clip_id
        frequency_sigma = self.frequency_sigma
        ori_clip_hw_feature = []
        for sigma_i, o_id in enumerate(select_ori_clip_id):
            _, b, c = clip_features_all[o_id].shape
            ori_clip_frequency = clip_features_all[o_id].permute(1, 2, 0).reshape(b, c, 16, 16)
            ori_clip_frequency = self.LFM(ori_clip_frequency, sigma=frequency_sigma[sigma_i])
            ori_clip_hw_feature.append(ori_clip_frequency.flatten(-2).permute(2,0,1))
        ori_clip_hw_feature = torch.concat(ori_clip_hw_feature, dim=0).transpose(0, 1)
        ori_clip_hw_feature = self.oriclip_mlp(ori_clip_hw_feature)


        clip_features = clip_features_all['final_cls_token']
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        clip_features = clip_features.unsqueeze(1)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        outputs["pred_logits"] = self.fusion_decoder(outputs["pred_logits"], ori_clip_hw_feature)

        outputs["pred_logits"] = outputs["pred_logits"].flatten(start_dim=0, end_dim=1)
        outputs["pred_logits"] = outputs["pred_logits"].unsqueeze(1)

        clip_features = clip_features.expand(-1, 100, -1)
        clip_features = clip_features.flatten(start_dim=0, end_dim=1)
        clip_features = clip_features.unsqueeze(1)


        outputs["pred_logits"] = self.lightweight_transformer(outputs["pred_logits"] + self.learnable_weight * clip_features)
        outputs["pred_logits"] = outputs["pred_logits"].reshape(-1, 100, 768)
        

        class_names = self.get_class_name_list(dataset_name)
        text_features = self.clip_adapter.get_text_features(class_names)
        outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
            text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
        )

        if self.training:
            if "aux_outputs" in outputs.keys():
                for i in range(len(outputs["aux_outputs"])):

                    outputs["aux_outputs"][i]["pred_logits"] = self.fusion_decoder(outputs["aux_outputs"][i]["pred_logits"], ori_clip_hw_feature)

                    outputs["aux_outputs"][i]["pred_logits"] = outputs["aux_outputs"][i]["pred_logits"].flatten(start_dim=0, end_dim=1)
                    outputs["aux_outputs"][i]["pred_logits"] = outputs["aux_outputs"][i]["pred_logits"].unsqueeze(1)

                    outputs["aux_outputs"][i]["pred_logits"] = self.lightweight_transformer(outputs["aux_outputs"][i]["pred_logits"]  + self.learnable_weight * clip_features)

                    outputs["aux_outputs"][i]["pred_logits"] = outputs["aux_outputs"][i]["pred_logits"].reshape(-1, 100, 768)

                    outputs["aux_outputs"][i]["pred_logits"] = self.clip_adapter.get_sim_logits(
                        text_features,
                        self.clip_adapter.normalize_feature(
                            outputs["aux_outputs"][i]["pred_logits"]
                        ),
                    )
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = image_size[0]
                width = image_size[1]
                mask_pred_result = sem_seg_postprocess(
                    mask_pred_result, image_size, height, width
                )
                image = input_per_image["image"].to(self.device)

                r, regions = self.semantic_inference(
                    mask_cls_result, mask_pred_result, image, class_names, dataset_name=dataset_name, clip_features_all=clip_features_all
                )

                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = sem_seg_postprocess(r, image_size, height, width)
                processed_results.append({"sem_seg": r})

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = self.panoptic_inference(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

            return processed_results


    def semantic_inference(self, mask_cls, mask_pred, image, class_names, dataset_name, clip_features_all=None):
        in_vocab_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()

        regions = None
        if self.clip_ensemble:
            clip_cls, regions, valid_flag = self.clip_adapter(
                image, class_names, mask_pred, normalize=True, clip_features_all=clip_features_all
            )
            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)

            if self.clip_ensemble_weight > 0:
                out_vocab_cls = in_vocab_cls.new_ones(in_vocab_cls.shape)
                out_vocab_cls[valid_flag] = clip_cls
                cls_results = torch.pow(in_vocab_cls, 1 - self.clip_ensemble_weight) * \
                            torch.pow(out_vocab_cls, self.clip_ensemble_weight)
                # we found log operation is beneficial for the a847 and pc459 dataset
                if dataset_name == 'ade20k_full_sem_seg_val' or dataset_name == 'pascal_context_459_sem_seg_val':
                    cls_results = cls_results.log()
                    is_void_prob = F.softmax(mask_cls, dim=-1)[..., -1:]
                    mask_cls_probs = torch.cat([
                        cls_results.softmax(-1) * (1.0 - is_void_prob),
                        is_void_prob], dim=-1)
                    mask_cls_results = torch.log(mask_cls_probs + 1e-8)
                    mask_cls = F.softmax(mask_cls_results, dim=-1)[..., :-1]
                else:
                    mask_cls = cls_results
            else:
                # only clip model predictions are used
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg, regions

    def get_class_name_list(self, dataset_name):
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        return class_names


@META_ARCH_REGISTRY.register()
class SCANDEMO(Mask2Former):
    """
    Main class for zero shot mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        clip_adapter: nn.Module,
        lightweight_transformer: nn.Module,
        LFM: nn.Module,
        fusion_decoder: nn.Module,
        oriclip_mlp: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        clip_ensemble: bool,
        clip_ensemble_weight: float,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        test_topk_per_image: int,
        select_ori_clip_id: list,
        frequency_sigma: list,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            clip_adapter: adapter for clip-based mask classification
            num_queries: int, number of queries
            panoptic_on: bool, whether to output panoptic segmentation prediction
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            semantic_on=semantic_on,
            instance_on=instance_on,
            panoptic_on=panoptic_on,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            test_topk_per_image=test_topk_per_image,
        )
        self.clip_adapter: ClipAdapter = clip_adapter

        self.clip_ensemble: bool = clip_ensemble
        self.clip_ensemble_weight: float = clip_ensemble_weight

        self.learnable_weight = nn.Parameter(torch.tensor([0.1]))
        self.lightweight_transformer = lightweight_transformer
        self.LFM = LFM
        self.fusion_decoder = fusion_decoder
        self.oriclip_mlp = oriclip_mlp

        self.select_ori_clip_id = select_ori_clip_id
        self.frequency_sigma = frequency_sigma

        self.openai_pixel_mean = torch.Tensor((0.48145466, 0.4578275, 0.40821073)).reshape(3, 1, 1) * 255
        self.openai_pixel_std = torch.Tensor((0.26862954, 0.26130258, 0.27577711)).reshape(3, 1, 1) * 255

    @classmethod
    def from_config(cls, cfg):
        init_kwargs = Mask2Former.from_config(cfg)
        text_templates = build_text_prompt(cfg.MODEL.CLIP_ADAPTER)

        clip_adapter = MaskFormerClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            text_templates,
            mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
            mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
            mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
            mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
            region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
            replace_ratio=cfg.MODEL.CLIP_ADAPTER.REPLACE_RATIO,
            replace_layer=cfg.MODEL.CLIP_ADAPTER.REPLACE_LAYER
        )
        init_kwargs["clip_adapter"] = clip_adapter
        init_kwargs["clip_ensemble"] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE
        init_kwargs[
            "clip_ensemble_weight"
        ] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT

        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=5)
        init_kwargs['lightweight_transformer'] = transformer_encoder
        init_kwargs['LFM'] = LFM(num_channels=1024)
        
        init_kwargs['fusion_decoder'] = CA(input_dim=768, num=5)
        init_kwargs['oriclip_mlp'] = MLP(1024, 768)

        init_kwargs['select_ori_clip_id'] = cfg.MODEL.SELECT_ORI_CLIP_ID
        init_kwargs['frequency_sigma'] = cfg.MODEL.FREQUENCY_SIGMA

        return init_kwargs

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]

        original_images = [(x - self.openai_pixel_mean.to(images[0].device)) / self.openai_pixel_std.to(images[0].device) for x in images]
        original_images = ImageList.from_tensors(original_images, self.size_divisibility)

        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        # use original clip
        clip_input_img = F.interpolate(original_images.tensor, [224, 224])
        clip_features_all = self.clip_adapter.original_clip(clip_input_img, get_embedding=True)
        select_ori_clip_id = self.select_ori_clip_id
        frequency_sigma = self.frequency_sigma
        ori_clip_hw_feature = []
        for sigma_i, o_id in enumerate(select_ori_clip_id):
            _, b, c = clip_features_all[o_id].shape
            ori_clip_frequency = clip_features_all[o_id].permute(1,2,0).reshape(b, c, 16, 16)
            ori_clip_frequency = self.LFM(ori_clip_frequency, sigma=frequency_sigma[sigma_i])
            ori_clip_hw_feature.append(ori_clip_frequency.flatten(-2).permute(2,0,1))
        ori_clip_hw_feature = torch.concat(ori_clip_hw_feature, dim=0).transpose(0, 1)
        ori_clip_hw_feature = self.oriclip_mlp(ori_clip_hw_feature)

        clip_features = clip_features_all['final_cls_token']
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        clip_features = clip_features.unsqueeze(1)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        outputs["pred_logits"] = self.fusion_decoder(outputs["pred_logits"], ori_clip_hw_feature)

        outputs["pred_logits"] = outputs["pred_logits"].flatten(start_dim=0, end_dim=1)
        outputs["pred_logits"] = outputs["pred_logits"].unsqueeze(1)

        clip_features = clip_features.expand(-1, 100, -1)
        clip_features = clip_features.flatten(start_dim=0, end_dim=1)
        clip_features = clip_features.unsqueeze(1)

        outputs["pred_logits"] = self.lightweight_transformer(outputs["pred_logits"] + self.learnable_weight * clip_features)
        outputs["pred_logits"] = outputs["pred_logits"].reshape(-1, 100, 768)

        class_names = batched_inputs[0]["class_names"]
        if len(class_names) == 1:
            # Because classification is performed in a 'contrastive' manner, adding others to represent other concepts
            class_names.append('others')
        text_features = self.clip_adapter.get_text_features(class_names)
        outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
            text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
        )
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        processed_results = []
        for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
            mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = image_size[0]
            width = image_size[1]
            mask_pred_result = sem_seg_postprocess(
                mask_pred_result, image_size, height, width
            )
            image = input_per_image["image"].to(self.device)

            r, regions = self.demo_inference(mask_cls_result, mask_pred_result, image, class_names, clip_features_all=clip_features_all)

            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = sem_seg_postprocess(r, image_size, height, width)
            processed_results.append({"sem_seg": r})

        return processed_results




    def demo_inference(self, mask_cls, mask_pred, image, class_names, clip_features_all=None):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]  # [100, 2]
        mask_pred = mask_pred.sigmoid()  # [100, 640, 854]

        regions = None
        if self.clip_ensemble:
            clip_cls, regions, valid_flag = self.clip_adapter(
                image, class_names, mask_pred, normalize=True, clip_features_all=clip_features_all
            )
            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)
            # softmax before index or after?
            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
            # self.clip_ensemble_weight = 0
            if self.clip_ensemble_weight > 0:
                map_back_clip_cls = mask_cls.new_ones(mask_cls.shape)
                map_back_clip_cls[valid_flag] = clip_cls
                mask_cls = torch.pow(mask_cls, 1 - self.clip_ensemble_weight) * \
                           torch.pow(map_back_clip_cls, self.clip_ensemble_weight)

            else:
                # only clip model predictions are used
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]  # mask_cls: [100, 2]

        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg, regions