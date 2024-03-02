# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
from collections import OrderedDict


# PATH to finetune clip model
clip_ckpt = torch.load('CS_CLIP.pt')

new_model = OrderedDict()
state_dict = clip_ckpt['state_dict']

for k, v in state_dict.items():
    if 'clip_model' in k:
        new_key = k.replace('module.clip_model.','')
        new_model[new_key] = v

# PATH to trained MaskFormer model
ovseg_model = torch.load('Seg_model.pth', 'cpu')

for k, v in new_model.items():
    new_k = 'clip_adapter.clip_model.' + k
    if new_k in ovseg_model['model'].keys():
        ovseg_model['model'][new_k] = v
    else:
        print(f'{new_k} does not exist in ckpt')
try:
    ovseg_model['model']['clip_adapter.clip_model.visual.mask_embedding'] = new_model['visual.mask_embedding']
    print('clip_ckpt has mask_embedding, remember to set MODEL.CLIP_ADAPTER.MASK_PROMPT_FWD True during evaluation')
except:
    print('clip_ckpt does not have mask_embedding, remember to set MODEL.CLIP_ADAPTER.MASK_PROMPT_FWD False during evaluation')

torch.save(ovseg_model, 'SCAN.pth')
