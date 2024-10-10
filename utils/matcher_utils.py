import torch
from copy import deepcopy
from LoFTR.src.loftr import LoFTR, default_cfg


def load_LoFTR(ckpt_path:str,temp_bug_fix:bool):
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = temp_bug_fix  # set to False when using the old ckpt
   
    LoFTR_model = LoFTR(config=_default_cfg)
    LoFTR_model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    LoFTR_model= LoFTR_model.eval().cuda()
    
    return LoFTR_model