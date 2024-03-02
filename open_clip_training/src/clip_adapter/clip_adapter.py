from typing import Tuple, Union, Callable, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from open_clip.factory import create_model_and_transforms
import copy

class ClipAdapter(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        model, preprocess_train, preprocess_val, preprocess_val_entire = create_model_and_transforms(
                                                    args.model,
                                                    args.pretrained,
                                                    precision=args.precision,
                                                    device=device,
                                                    jit=args.torchscript,
                                                    force_quick_gelu=args.force_quick_gelu,
                                                    force_custom_text=args.force_custom_text,
                                                    force_patch_dropout=args.force_patch_dropout,
                                                    force_image_size=args.force_image_size,
                                                    image_mean=args.image_mean,
                                                    image_std=args.image_std,
                                                    image_interpolation=args.image_interpolation,
                                                    image_resize_mode=args.image_resize_mode,  # only effective for inference
                                                    aug_cfg=args.aug_cfg,
                                                    pretrained_image=args.pretrained_image,
                                                    output_dict=True,
                                                    with_mask=args.with_mask,
                                                    mask_emb_depth=args.mask_emb_depth
                                                )
        
        self.clip_model = model
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val
        self.preprocess_val_entire = preprocess_val_entire

        self.original_clip_visual = copy.deepcopy(model.visual)
        for _, param in self.original_clip_visual.named_parameters():
            param.requires_grad = False
    
    def forward(self, original_image, image, text, mask=None):
        if image is None:
            return self.encode_text(text)
        elif text is None:
            ori_image_features = self.original_clip_visual(original_image, get_embedding=True)
            image_features = self.clip_model.encode_image(image, ori_image_features=ori_image_features, mask=mask)  # [32, 768]

            image_features = F.normalize(image_features, dim=-1)  # [32, 768]
            return {'image_features': image_features}
        
        if mask is None:
            ori_image_features = self.original_clip_visual(original_image, get_embedding=True)
            image_features = self.clip_model.encode_image(image, ori_image_features=ori_image_features)  # [32, 768]
        else:
            ori_image_features = self.original_clip_visual(original_image, get_embedding=True)
            image_features = self.clip_model.encode_image(image, ori_image_features=ori_image_features, mask=mask)  # [32, 768]

        image_features = F.normalize(image_features, dim=-1)  # [32, 768]

        text_features = self.clip_model.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)  # [32, 768]

        # return image_features, text_features, self.clip_model.logit_scale.exp()
        out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.clip_model.logit_scale.exp()
            }
        if self.clip_model.logit_bias is not None:
            out_dict['logit_bias'] = self.logit_bias
        return out_dict