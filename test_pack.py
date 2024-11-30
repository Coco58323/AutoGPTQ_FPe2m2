import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import transformers
def e2m2_map(x):
    if x==0:
        return 0
    if x==0.5:
        return 1
    if x==1:
        return 2
    if x==1.5:
        return 3
    if x==2:
        return 4
    if x==2.5:
        return 5
    if x==3:
        return 6
    if x==3.5:
        return 7
    if x==4:
        return 8
    if x==5:
        return 9
    if x==6:
        return 10
    if x==7:
        return 11
    if x==8:
        return 12
    if x==10:
        return 13
    if x==12:
        return 14
    if x==14:
        return 15
    print(x)
    return ValueError("Invalid input")
def e2m2_map_inv(x):
    if x==0:
        return 0
    if x==1:
        return 0.5
    if x==2:
        return 1
    if x==3:
        return 1.5
    if x==4:
        return 2
    if x==5:
        return 2.5
    if x==6:
        return 3
    if x==7:
        return 3.5
    if x==8:
        return 4
    if x==9:
        return 5
    if x==10:
        return 6
    if x==11:
        return 7
    if x==12:
        return 8
    if x==13:
        return 10
    if x==14:
        return 12
    if x==15:
        return 14
    # print(x.dtype)
    return ValueError("Invalid input")

# sparse_dict = {
#     0: 0, 0.5: 1, 1: 2, 1.5: 3, 2: 4, 2.5: 5, 
#     3: 6, 3.5: 7, 4: 8, 5: 9, 6: 10, 7: 11, 
#     8: 12, 10: 13, 12: 14, 14: 15
# }
sparse_dict = {
    0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 10:9, 12:10, 14:11, 16:12, 20:13, 24:14, 28:15
}
e2m2_keys = torch.tensor(list(sparse_dict.keys()), dtype=torch.float32)
e2m2_values = torch.tensor(list(sparse_dict.values()), dtype=torch.int64)
inverse_table = torch.tensor([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14], dtype=torch.float32)


def get_scale(input, bits, mantissa_bit, bias):
        M = mantissa_bit
        E = bits - 1 - M
        maxval = (2 - 2 ** (-M)) * 2 ** (
                2**E - 1 - bias
            )
        minval = -maxval
        input = input.clamp(minval, maxval)
        input_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(input)) + bias)).detach(), 1.0)
        return input, 2.0 ** (input_log_scales - M - bias)
def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x
def sym_quant_fpe2m2(x,scale=None):
    dtype=x.dtype
    if scale is None:
        scale = x.abs().amax(dim=-1,keepdim=True) /14
    scale = scale.to(x.device)
    norm = x / scale
    norm=norm.clamp(min=-14,max=14)
    norm_e2m2 = norm
    w, w_scale = get_scale(norm_e2m2,5,2,0)
    w=round_ste(w/w_scale)
    w_sim=w.mul(w_scale)
    return w_sim

def sym_quant_fpe2m2_fake(x,scale=None,zero=None,power_scale=None):
    dtype=x.dtype
    if scale is None:
        scale = x.abs().amax(dim=-1,keepdim=True) /14
    scale = scale.to(x.device)
    norm = x / scale
    norm=norm.clamp(min=-14,max=14)
    norm_e2m2 = norm
    w, w_scale = get_scale(norm_e2m2,5,2,0)
    w=round_ste(w/w_scale)
    w_sim=w.mul(w_scale)
    w_e2m2 = w_sim.div(14)
    w = w_e2m2
    w_sim = w.mul(scale).to(dtype=dtype)
    return w_sim
class WeightPack:
    def __init__(self, infeatures, outfeatures, bits=8):
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.scales = torch.zeros((infeatures, outfeatures), dtype=torch.float32)
        self.qweight = None
        self.sign_map = None
        self.weight = None
        self.inverse_table = inverse_table
        self.bias = None
        self.wf = torch.tensor(list(range(0, 32, 4)), dtype=torch.int32).unsqueeze(0)
        self.wf_sign = torch.tensor(list(range(0, 32)), dtype=torch.int32).unsqueeze(0)
    def pack(self, linear, scales=None, zero=None, g_idx=None):
        # self.weight = linear.weight.data.clone()
        W = linear.weight.data.clone()
        if scales is None:
            scales = W.abs().amax(dim=-1, keepdim=True)/14
        scales = scales.contiguous()
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=linear.weight.dtype)
        quant_weight = sym_quant_fpe2m2(W, scales)
        sign_map = torch.sign(quant_weight)
        sign_map = torch.where(sign_map == 1, 0, 1)
        quant_weight = quant_weight.abs()
        quant_weight = quant_weight * 2
        # apply element-wise e2m2 map func to quant_weight
        # quant_weight = quant_weight.apply_(e2m2_map)
        indices = torch.bucketize(quant_weight, e2m2_keys)
        quant_weight = torch.take(e2m2_values, indices)
        intweight = []

        for idx in range(self.infeatures):
            intweight.append(quant_weight[:, idx].to(torch.int)[
                    :, None
                ]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        sign_map = sign_map.t().contiguous()
        sign_map = sign_map.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        q_sign_map = np.zeros((intweight.shape[0] // 32, intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            for j in range(i, i + (32 // self.bits)):
                t = intweight[j] & ((2**self.bits - 1)<<self.bits)
                # check if any non-zero bits are in the upper bits of the tensor
                check = (t!=0).any()
                if check:
                    raise ValueError("Non-zero bits in upper bits")
                qweight[row] |= intweight[j] << (self.bits * (j - i))
            i += 32 // self.bits
            row += 1
        row = 0
        i = 0
        while row < q_sign_map.shape[0]:
            for j in range(i, i + 32):
                q_sign_map[row] |= sign_map[j] << (j - i)
            i += 32
            row += 1

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)
        q_sign_map = q_sign_map.astype(np.int32)
        self.sign_map = torch.from_numpy(q_sign_map)
        self.scales = scales.clone().to(dtype=linear.weight.dtype).T
    def unpack(self):
        if self.weight is not None:
            return
        self.wf = self.wf.to(self.qweight.device)
        self.inverse_table = self.inverse_table.to(self.qweight.device)
        intweight = torch.bitwise_right_shift(
            torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
            self.wf.unsqueeze(-1),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        intweight = torch.bitwise_and(intweight, (2**self.bits) - 1).to(torch.long)
        # intweight = intweight.apply_(e2m2_map_inv)
        intweight = torch.take(self.inverse_table, intweight)
        intweight = intweight.reshape(self.infeatures, self.outfeatures).T
        bit_map = torch.bitwise_right_shift(
            torch.unsqueeze(self.sign_map, 1).expand(-1, 32, -1),
            self.wf_sign.unsqueeze(-1),
        )
        bit_map = torch.bitwise_and(bit_map, 1).to(torch.uint32)
        bit_map = bit_map.reshape(self.infeatures, self.outfeatures)
        bit_map = bit_map.t()
        # self.weight = intweight * (self.scales.T)
        self.weight = intweight
        self.weight = self.weight.to(self.wf.device)
        bit_map = bit_map.to(self.wf.device)
        self.weight = torch.where(bit_map == 1, -self.weight, self.weight)
        
weight = nn.Linear(4096, 4096)
weight.weight.data = sym_quant_fpe2m2_fake(weight.weight.data)
weight_pack = WeightPack(4096, 4096, 4)
weight_pack.pack(weight)
weight_pack.unpack()
weight1 = weight_pack.weight
weight2 = weight
assert torch.allclose(weight1, weight2, atol=1e-5)
weight1 = torch.randn(4096, 4096)
activation = torch.randn(4096, 4096)
activation[:, 0] = torch.ones(4096)*20
activation[:, 1] = torch.ones(4096)*30
qmax = torch.finfo(torch.float8_e4m3fn).max-1
weight1 = weight1.clamp(-qmax, qmax)
activation = activation.clamp(-qmax, qmax)

output1 = torch.matmul(activation, weight1.t())

weight1 = weight1.to(torch.float8_e4m3fn).to(torch.float)
scales = activation.abs().amax(dim=-1, keepdim=True) / qmax
activation = activation / scales
activation = activation.to(torch.float8_e4m3fn).to(torch.float)
output2 = torch.matmul(activation, weight1.t())
output2 = output2 * scales

weight1 = weight1.to(torch.float8_e4m3fn)
activation = activation.to(torch.float8_e4m3fn)
# todo: replace gemm_e4m3fn with the correct function
otuput3 = gemm_e4m3fn(activation, weight1.t())

assert torch.allclose(output2, otuput3, atol=1e-5)