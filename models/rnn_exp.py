import approximate
import torch
from torch.nn import functional as F


def lstm_cell(xt, ht_1, ct_1, w_ih, w_hh, b_ih, b_hh):
    gates = F.linear(xt, w_ih, b_ih) + F.linear(ht_1, w_hh, b_hh)
    it, ft, gt, ot = gates.chunk(4, 1)
    it = torch.sigmoid(it)
    ft = torch.sigmoid(ft)
    gt = torch.tanh(gt)
    ot = torch.sigmoid(ot)
    ct = (ft * ct_1) + (it * gt)
    ht = ot * torch.tanh(ct)
    return ht, ct

def lstm_cell_post_appr1_bf16(xt, ht_1, ct_1, w_ih, w_hh, b_ih, b_hh):
    gates = F.linear(xt, w_ih, b_ih) + F.linear(ht_1, w_hh, b_hh)
    gates = gates.to(torch.bfloat16)
    ct_1 = ct_1.to(torch.bfloat16)
    it, ft, gt, ot = gates.chunk(4, 1)
    it = approximate.appro_sigmoid(it)
    ft = approximate.appro_sigmoid(ft)
    gt = approximate.appro_tanh(gt)
    ot = approximate.appro_sigmoid(ot)
    ct = (ft * ct_1) + (it * gt)
    ht = ot * approximate.appro_tanh(ct)
    ht = ht.to(torch.float32)
    ct = ct.to(torch.float32)
    return ht, ct

def lstm_cell_post_appr1(xt, ht_1, ct_1, w_ih, w_hh, b_ih, b_hh):
    gates = F.linear(xt, w_ih, b_ih) + F.linear(ht_1, w_hh, b_hh)
    it, ft, gt, ot = gates.chunk(4, 1)
    it = approximate.appro_sigmoid(it)
    ft = approximate.appro_sigmoid(ft)
    gt = approximate.appro_tanh(gt)
    ot = approximate.appro_sigmoid(ot)
    ct = (ft * ct_1) + (it * gt)
    ht = ot * approximate.appro_tanh(ct)
    return ht, ct

def lstm_cell_post_appr2(xt, ht_1, ct_1, w_ih, w_hh, b_ih, b_hh):
    gates = F.linear(xt, w_ih, b_ih) + F.linear(ht_1, w_hh, b_hh)
    it, ft, gt, ot = gates.chunk(4, 1)
    it = torch.sigmoid(it)
    ft = torch.sigmoid(ft)
    gt = approximate.appro_tanh(gt)
    ot = torch.sigmoid(ot)
    ct = (ft * ct_1) + (it * gt)
    ht = ot * approximate.appro_tanh(ct)
    return ht, ct

def lstm_cell_post_appr3(xt, ht_1, ct_1, w_ih, w_hh, b_ih, b_hh):
    gates = F.linear(xt, w_ih, b_ih) + F.linear(ht_1, w_hh, b_hh)
    it, ft, gt, ot = gates.chunk(4, 1)
    it = approximate.appro_sigmoid(it)
    ft = approximate.appro_sigmoid(ft)
    gt = torch.tanh(gt)
    ot = approximate.appro_sigmoid(ot)
    ct = (ft * ct_1) + (it * gt)
    ht = ot * torch.tanh(ct)
    return ht, ct

def lstm_cell_post_bf16(xt, ht_1, ct_1, w_ih, w_hh, b_ih, b_hh):
    gates = F.linear(xt, w_ih, b_ih) + F.linear(ht_1, w_hh, b_hh)
    gates = gates.to(torch.bfloat16)
    ct_1 = ct_1.to(torch.bfloat16)
    it, ft, gt, ot = gates.chunk(4, 1)
    it = torch.sigmoid(it)
    ft = torch.sigmoid(ft)
    gt = torch.tanh(gt)
    ot = torch.sigmoid(ot)
    ct = (ft * ct_1) + (it * gt)
    ht = ot * torch.tanh(ct)
    ht = ht.to(torch.float32)
    ct = ct.to(torch.float32)
    return ht, ct

def lstm_cell_post_int8(xt, ht_1, ct_1, w_ih, w_hh, b_ih, b_hh):
    gates = F.linear(xt, w_ih, b_ih) + F.linear(ht_1, w_hh, b_hh)
    it, ft, gt, ot = gates.chunk(4, 1)
    it = torch.sigmoid(it)
    ft = torch.sigmoid(ft)
    gt = torch.tanh(gt)
    ot = torch.sigmoid(ot)
    ft = torch.clamp((ft*127).round(), -127, 127) / 127
    ct_1 = torch.clamp((ct_1*127).round(), -127, 127) / 127
    it = torch.clamp((it*127).round(), -127, 127) / 127
    gt = torch.clamp((gt*127).round(), -127, 127) / 127
    ct = (ft * ct_1) + (it * gt)
    ht = ot * torch.tanh(ct)
    return ht, ct

