# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


def min_max_normalize(matrix,min_vals=None,max_vals=None,dim=1):
    if min_vals is None:
        min_vals, _ = torch.min(matrix, dim=dim, keepdim=True)
    if max_vals is None:
        max_vals, _ = torch.max(matrix, dim=dim, keepdim=True)
    normalized_matrix = (matrix - min_vals) / (max_vals - min_vals)
    return normalized_matrix

def batch_get_rand_negqr(candi_vec,each_select):
    bs, negn, qn, fd = candi_vec.shape[0], candi_vec.shape[1], candi_vec.shape[2], candi_vec.shape[-1]
    indices = torch.randint(0, qn, (bs, negn, each_select)).to(candi_vec.device)
    return indices

def batch_get_difficult_negqr(candi_vec, lan_emb, each_select):
    lan_similarity = torch.einsum('bnvd, bd -> bnv',candi_vec,lan_emb.squeeze(1))
    difficulty = min_max_normalize(lan_similarity,dim=2)
    select_score = difficulty
    value, res_indices = torch.topk(select_score,k=each_select,dim=-1)
    return res_indices

def batch_get_hq_negqr(candi_vec, lan_emb, each_select=4):
    bs, negn, qn, fd = candi_vec.shape[0], candi_vec.shape[1], candi_vec.shape[2], candi_vec.shape[-1] # bs,(bs-1),qn,fd: 16,15,20,512
    norm_candi = F.normalize(candi_vec, p=2, dim=-1)
    norm_all = F.normalize(candi_vec.view(bs, negn*qn, fd), p=2, dim=-1)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        self_similarity = torch.matmul(norm_candi.unsqueeze(3), norm_all.unsqueeze(1).unsqueeze(2).transpose(-1,-2)).squeeze().to(torch.float32)
    uniqueness = torch.ones(bs, negn, qn).to(candi_vec.device) 
    max_vals = torch.ones((bs, 1)).to(candi_vec.device)
    self_sim_norm = min_max_normalize(-self_similarity,(max_vals * -1).unsqueeze(2).unsqueeze(3),max_vals.unsqueeze(2).unsqueeze(3))
    lan_similarity = torch.einsum('bnvd, bd -> bnv',candi_vec,lan_emb.squeeze(1))
    difficulty = min_max_normalize(lan_similarity,dim=2)
    select_score = uniqueness * (difficulty)
    # select_score = uniqueness + difficulty
    # select_score = uniqueness * (difficulty**2)

    for i in range(each_select):
        value, indices = torch.max(select_score,dim=-1)  
        mul_indices = indices + (torch.arange(negn).unsqueeze(0).expand(bs,negn).to(candi_vec.device)*qn)
        mul_indices = mul_indices.unsqueeze(1).expand(bs,negn,negn).unsqueeze(2).expand(bs,negn,qn,negn)
        indices = indices.unsqueeze(2)
        query_value = self_sim_norm.gather(3,mul_indices.to(self_sim_norm.device))
        query_value = torch.min(query_value,dim=-1)[0]
        uniqueness, _ = torch.min(torch.stack([uniqueness, query_value]), dim=0) 
        select_score = uniqueness * (difficulty)
        # select_score = uniqueness + difficulty
        # select_score = uniqueness * (difficulty**2)
        if i == 0:
            res_indices = indices
        else:
            res_indices = torch.cat([res_indices,indices],dim=2)
    return res_indices


def get_contrast(vis_emb, lan_emb, stat_sim_dict=None, each_select=4):
    bs, qn, fd = vis_emb.shape
    vis_emb_orig = vis_emb.clone()
    vis_emb_bs = vis_emb_orig.unsqueeze(0).expand(bs,bs,qn,fd)
    sim_map = torch.einsum('avd, bqd -> baqv',vis_emb,lan_emb)
    max_sims, indices = sim_map.topk(k=qn, dim=-1, largest=True, sorted=True)
    indices = indices.squeeze(2)
    n_negqn = indices.shape[-1]
    neg_queries = vis_emb_bs.gather(2,indices.unsqueeze(3).expand(bs,bs,n_negqn,fd)).to(vis_emb.device)
    # pos_queries = neg_queries[:,:,0,:].masked_select(torch.eye(bs).bool().unsqueeze(2).expand(bs,bs,fd).to((neg_queries.device))).contiguous().view(bs,1,fd)
    neg_queries = neg_queries.masked_select(~torch.eye(bs).bool().unsqueeze(2).unsqueeze(3).expand(bs,bs,n_negqn,fd).to(neg_queries.device)).contiguous().view(bs,(bs-1),n_negqn,fd)
    candidate_queries = neg_queries
    hq_indices = batch_get_hq_negqr(candidate_queries, lan_emb, each_select=each_select)
    hq_negqr = candidate_queries.gather(2,hq_indices.unsqueeze(3).expand(bs,(bs-1),hq_indices.shape[2],fd)).to(candidate_queries.device).view(bs,(bs-1)*hq_indices.shape[2],fd)
    sim_neg_map = torch.einsum('bkd, byd -> byk', hq_negqr, lan_emb)
    sim_neg_map = sim_neg_map.squeeze(1)
    max_sims = max_sims.squeeze(2)
    max_sim_0 = max_sims[...,0]
    max_sim_0_pos = max_sim_0.masked_select(torch.eye(bs).bool().to(max_sim_0.device)).contiguous().view(bs,1)
    if stat_sim_dict is not None:
        pos_sim_mean = torch.mean(max_sim_0.masked_select(torch.eye(bs).bool().to(max_sim_0.device)).contiguous().view(bs))
        neg_sim_top1_mean = torch.mean(max_sim_0.masked_select(~torch.eye(bs).bool().to(max_sim_0.device)).contiguous().view(bs,bs-1))
        sim_hq_mean = torch.mean(sim_neg_map)
        stat_sim_dict["sim_hq_mean"] += sim_hq_mean.item()
        stat_sim_dict["pos_sim_mean"] += pos_sim_mean.item()
        stat_sim_dict["neg_sim_top1_mean"] += neg_sim_top1_mean.item()
        stat_sim_dict["num"] += 1

    new_logits = torch.cat([max_sim_0_pos,sim_neg_map],dim=1) 
    target_pred = torch.Tensor([0 for _ in range(bs)]).to(torch.int64).to(max_sim_0.device)
    loss_contra = nn.CrossEntropyLoss(reduction="mean")(new_logits, target_pred)
    return loss_contra, stat_sim_dict


def get_prediction(vis_emb, lan_emb):
    sim_map = torch.einsum('bkd, byd -> byk', vis_emb, lan_emb)
    maxval, v = sim_map.max(dim=2, keepdim=True)
    predictions = torch.zeros_like(sim_map).to(sim_map.device).scatter(2,v.expand(sim_map.shape), 1).bool()
    return predictions


class WeakREShead(nn.Module):
    def __init__(self, __C):
        super(WeakREShead, self).__init__()
        self.each_select = __C.EACH_SELECT
    def forward(self, vis_fs,lan_fs,stat_sim_dict=None):
        if self.training:
            loss_contra, stat_sim_dict = get_contrast(vis_fs, lan_fs, stat_sim_dict, each_select=self.each_select)
            return loss_contra, stat_sim_dict
        else:
            predictions = get_prediction(vis_fs, lan_fs)
            return predictions