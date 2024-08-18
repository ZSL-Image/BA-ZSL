# from matplotlib.cbook import flatten
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import os
from torch.nn.functional import dropout
from models.modeling import utils
from models.modeling.backbone_vit.vit_model import vit_base_patch16_224 as create_model

from os.path import join
import pickle
import numpy as np
from torch.autograd import Variable
import torch.nn.init as init
Norm =nn.LayerNorm


def trunc_normal_(tensor, mean=0, std=.01):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class SimpleReasoning(nn.Module):
    def __init__(self, np):
        super(SimpleReasoning, self).__init__()
        self.hidden_dim= np
        self.fc1 = nn.Linear(np, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, np)
        self.avgpool = nn.AdaptiveMaxPool1d(1)
        self.act=nn.GELU()

    def forward(self, x):
        x_1 = self.fc1(self.avgpool(x).flatten(1)) 
        x_1 = self.act(x_1)
        x_1 = torch.sigmoid(self.fc2(x_1)).unsqueeze(-1)
        x_1 = x_1*x
        return x_1
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
        bound1 = 1 / math.sqrt(fan_in1)
        nn.init.uniform_(self.fc1.bias, -bound1, bound1)
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2.weight)
        bound2 = 1 / math.sqrt(fan_in2)
        nn.init.uniform_(self.fc2.bias, -bound2, bound2)


class AnyAttention(nn.Module):
    def __init__(self, dim, qkv_bias=False):
        super(AnyAttention, self).__init__()
        self.norm_q, self.norm_k, self.norm_v = Norm(dim), Norm(dim), Norm(dim)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = dim ** (-0.5)
        self.act=nn.ReLU()
        self.proj = nn.Linear(dim, dim)
    def get_qkv(self, q, k, v):
        q, k, v = self.norm_q(q), self.norm_k(k), self.norm_v(v)
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        return q, k, v
    def forward(self, q=None, k=None, v=None):
        q, k, v = self.get_qkv(q, k, v)
        attn = torch.einsum("b q c, b k c -> b q k", q, k)
        attn = self.act(attn)
        attn *= self.scale
        attn_mask = F.softmax(attn, dim=-1)
        out = torch.einsum("b q k, b k c -> b q c", attn_mask, v.float())
        out = self.proj(out)
        return attn, out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(hidden_features) or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
        bound1 = 1 / math.sqrt(fan_in1)
        nn.init.uniform_(self.fc1.bias, -bound1, bound1)
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2.weight)
        bound2 = 1 / math.sqrt(fan_in2)
        nn.init.uniform_(self.fc2.bias, -bound2, bound2)



class Block(nn.Module):
    def __init__(self, dim, ffn_exp=4, num_parts=0):
        super(Block, self).__init__()
        self.dec_attn = AnyAttention(dim, True)
        self.ffn = Mlp(dim, hidden_features=dim * ffn_exp, act_layer=nn.GELU, norm_layer=Norm)
        self.enc_attn = AnyAttention(dim, True)
        self.attribute_distinction = SimpleReasoning(num_parts)
        self.maxpool1d = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, parts=None):
        x = rearrange(x, "b c p -> b p c")
        # MGSFL    Commenting out the ADWM and SGCMI codes is the ablation experiment for MGSFL.
        # ADWM    Commenting out this code is the ablation experiment on ADWM.
        parts1 = self.attribute_distinction(parts)
        parts_out = parts1 + parts
        parts = parts_out

        # Semantic-Guided Cross-Modal Interaction (SGCMI)
        attn_0, attn_out = self.enc_attn(q=parts, k=x, v=x)
        attn_0 = self.maxpool1d(attn_0).flatten(1)
        parts_in= parts + attn_out

        # IVFL
        # VIEM Commenting out this code is the ablation experiment on VIEM.
        feats1 = x + self.ffn(x)

        # Visual-Guided Cross-Modal Interaction (VGCMI)
        attn_, feats = self.dec_attn(q=feats1, k=parts_in, v=parts_in)
        feats = x + feats

        feats = rearrange(feats, "b p c -> b c p")
        return feats, attn_0
    

class MGFIN(nn.Module):
    def __init__(self, basenet, c, w, h,
                 attritube_num, cls_num, ucls_num, w2v,
                 scale=20.0, device=None):

        super(MGFIN, self).__init__()
        self.attritube_num = attritube_num
        self.feat_channel = c
        self.feat_wh = w * h
        self.batch =10
        self.cls_num= cls_num
        self.ucls_num = ucls_num
        self.scls_num = cls_num - ucls_num
        self.w2v_att = torch.from_numpy(w2v).float().to(device)
        self.mask = torch.zeros(self.cls_num).to(device)
        self.mask[:self.scls_num] = -1
        self.mask[-self.ucls_num:] = 1
        self.W = nn.Parameter(trunc_normal_(torch.empty(self.w2v_att.shape[1], self.feat_channel)),
                              requires_grad=True)
        self.V = nn.Parameter(trunc_normal_(torch.empty(self.feat_channel, self.attritube_num)),
                              requires_grad=True)
        self.V2 = nn.Parameter(trunc_normal_(torch.empty(self.feat_channel, self.attritube_num)),
                              requires_grad=True)
        assert self.w2v_att.shape[0] == self.attritube_num
        if scale<=0:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)

        self.backbone_patch = nn.Sequential(*list(basenet.children()))[0]
        self.backbone_drop= nn.Sequential(*list(basenet.children()))[1]
        self.backbone = nn.Sequential(*list(basenet.children()))[2]

        self.drop_path = 0.1

        self.cls_token = basenet.cls_token
        self.pos_embed = basenet.pos_embed

        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.CLS_loss = nn.CrossEntropyLoss()
        self.Reg_loss = nn.MSELoss()

        self.blocks = Block(self.feat_channel,
                  ffn_exp=4,
                  num_parts=self.attritube_num)


    def compute_score(self, gs_feat,seen_att,att_all):
        gs_feat = gs_feat.view(self.batch, -1)
        gs_feat_norm = torch.norm(gs_feat, p=2, dim = 1).unsqueeze(1).expand_as(gs_feat)
        gs_feat_normalized = gs_feat.div(gs_feat_norm + 1e-5)
        temp_norm = torch.norm(att_all, p=2, dim=1).unsqueeze(1).expand_as(att_all)
        seen_att_normalized = att_all.div(temp_norm + 1e-5)
        score_o = torch.einsum('bd,nd->bn', gs_feat_normalized, seen_att_normalized)  

        d, _ = seen_att.shape
        score_o = score_o*self.scale
        if d == self.cls_num:
            score = score_o
        if d == self.scls_num:
            score = score_o[:, :d]
            if self.training:
                return score
        if d == self.ucls_num:
            score = score_o[:, -d:]
        return score
    
    def forward(self, x, att=None, label=None, seen_att=None, att_all=None, score_local=1.0):
        self.batch = x.shape[0]
        parts = torch.einsum('lw,wv->lv', self.w2v_att, self.W)
        parts = parts.expand(self.batch, -1, -1)
        
        patches = self.backbone_patch(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        patches = torch.cat((cls_token, patches), dim=1) 
        feats_0 = self.backbone_drop(patches + self.pos_embed)
        feats_0 = self.backbone(feats_0)
        feats_in = feats_0[:, 1:, :]
        feats_g = feats_0[:, 0, :]
        feats, att_mask = self.blocks(feats_in.transpose(1,2), parts=parts)    # Commenting out this code will result in the baseline in the ablation experiment. Directly use the features extracted from ViT for classification.
        
        out = self.avgpool1d(feats.view(self.batch, self.feat_channel, -1)).view(self.batch, -1)
        out1 = torch.einsum('bc,cd->bd', out, self.V)
        score = self.compute_score(out1, seen_att, att_all)

        out_g = torch.einsum('bc,cd->bd', feats_g, self.V2)    # Commenting out this code is the ablation experiment on GVF. And the corresponding other codes need to be changed.
        score_g = self.compute_score(out_g, seen_att, att_all)    # Commenting out this code is the ablation experiment on GVF.

        if not self.training:
            return score_local*score + (1.0-score_local)*score_g    # Commenting out this code is the ablation experiment on GVF. Only return 'score'.
            # return score
        
        Lmse = self.Reg_loss(att_mask, att)    # Commenting out this code is the ablation experiment on L_MSE. And the corresponding other codes need to be commented out.
        Lcls = self.CLS_loss(score, label)
        Lcls_g = self.CLS_loss(score_g, label)    # Commenting out this code is the ablation experiment on GVF.
        scale = self.scale.item()
        loss_dict = {
            'Reg_loss': Lmse,    # Commenting out this code is the ablation experiment on L_MSE.
            'Cls_loss': Lcls,
            'scale': scale,
            'Clsg_loss': Lcls_g    # Commenting out this code is the ablation experiment on GVF.
        }

        return loss_dict


def build_model(cfg):
    dataset_name = cfg.DATASETS.NAME
    info = utils.get_attributes_info(dataset_name)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    c,w,h = 768, 14, 14
    scale = cfg.MODEL.SCALE 
    vit_model = create_model(num_classes=-1)
    vit_model_path = "./pretrain_model_vit/vit_base_patch16_224.pth"
    weights_dict = torch.load(vit_model_path)
    del_keys = ['head.weight', 'head.bias'] if vit_model.has_logits \
        else ['head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    vit_model.load_state_dict(weights_dict, strict=False)
    w2v_file = dataset_name+"_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device(cfg.MODEL.DEVICE)

    return MGFIN(basenet=vit_model,
                  c=c,w=w,h=h,scale=scale,
                  attritube_num=attritube_num, w2v=w2v,
                  cls_num=cls_num, ucls_num=ucls_num,
                  device=device)
