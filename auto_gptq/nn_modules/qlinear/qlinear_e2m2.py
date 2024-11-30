from logging import getLogger

import numpy as np
import torch
import torch.nn as nn

sparse_dict = {
    0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 10:9, 12:10, 14:11, 16:12, 20:13, 24:14, 28:15
}
e2m2_keys = torch.tensor(list(sparse_dict.keys()), dtype=torch.int32)
e2m2_values = torch.tensor(list(sparse_dict.values()), dtype=torch.int32)
inverse_table = torch.tensor([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14], dtype=torch.float32)
def get_scale(input, bits, mantissa_bit):
        M = mantissa_bit
        E = bits - 1 - M
        maxval = (2 - 2 ** (-M)) * 2 ** (
                2**E - 1 
            )
        minval = -maxval
        input = input.clamp(minval, maxval)
        input_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(input)) )).detach(), 1.0)
        return input, 2.0 ** (input_log_scales - M)


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def sym_quant_fpe2m2(x,scale=None,qmax=28,bits=5,mb=2):
    dtype=x.dtype
    if scale is None:
        scale = x.abs().amax(dim=-1,keepdim=True)/(qmax/2)
    scale = scale.to(x.device)
    x = x.div(scale)
    x=x.clamp(min=-qmax/2,max=qmax/2)
    pot, v_step = get_scale(x,bits,mb)
    x=round_ste(pot/v_step).mul(v_step)
    return x

logger = getLogger(__name__)

try:
    _autogptq_cuda_available = True
except ImportError:
    logger.warning("CUDA extension not installed.")
    autogptq_cuda_256 = None
    autogptq_cuda_64 = None
    _autogptq_cuda_available = False


class QuantLinear(nn.Module):
    QUANT_TYPE = "cuda"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        name=None,
        use_cuda_fp16=False,
        kernel_switch_threshold=128,
        trainable=False,
        weight_dtype=torch.bfloat16,
    ):
        super().__init__()
        global _autogptq_cuda_available
        if trainable:
            _autogptq_cuda_available = False

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.group_size = group_size if group_size != -1 else infeatures

        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 8, outfeatures), dtype=torch.int32),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (1, outfeatures),
                dtype=torch.bfloat16,
            ),
        )
        self.register_buffer(
            "sign_map",
            torch.zeros((infeatures // 32, outfeatures), dtype=torch.int32),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=weight_dtype))
        else:
            self.bias = None

        self.wf = torch.tensor(list(range(0, 32, 4)), dtype=torch.int32).unsqueeze(0)
        self.wf_sign = torch.tensor(list(range(0, 32)), dtype=torch.int32).unsqueeze(0)
        self.weight = None
        self.ready=False
        self.inverse_table = inverse_table
        self.e2m2_keys = e2m2_keys
        self.e2m2_values = e2m2_values
        self.name = name
    def post_init(self):
        pass

    def pack(self, linear, scales=None, zero=None, g_idx=None):
        W = linear.weight.data.clone()
        scales = scales.to(linear.weight.dtype)
        if scales is None:
            raise ValueError("scales is None")
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=linear.weight.dtype)
        quant_weight = sym_quant_fpe2m2(W, scale=scales)
        self.compare_weight = quant_weight*scales
        sign_map = torch.sign(quant_weight)
        sign_map = torch.where(sign_map == 1, 0, 1)
        quant_weight = quant_weight.abs()
        quant_weight = quant_weight * 2
        quant_weight = quant_weight.to(torch.int32)
        indices = torch.bucketize(quant_weight, self.e2m2_keys)
        quant_weight = torch.take(self.e2m2_values, indices)
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
        qweight = np.zeros((intweight.shape[0] // 8, intweight.shape[1]), dtype=np.uint32)
        q_sign_map = np.zeros((intweight.shape[0] // 32, intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            for j in range(i, i + (8)):
                qweight[row] |= intweight[j] << (4* (j - i))
            i += 8
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
        self.scales = scales.clone().T
    def unpack(self):
        if self.weight is not None:
            return self.weight
        self.wf = self.wf.to(self.qweight.device)
        self.inverse_table = self.inverse_table.to(self.qweight.device)
        intweight = torch.bitwise_right_shift(
            torch.unsqueeze(self.qweight, 1).expand(-1, 8, -1),
            self.wf.unsqueeze(-1),
        ).to(torch.int8)
        intweight = torch.bitwise_and(intweight, (2**4) - 1).to(torch.long)
        intweight = torch.take(self.inverse_table, intweight)
        intweight = intweight.reshape(self.infeatures, self.outfeatures).T
        self.wf_sign = self.wf_sign.to(self.qweight.device)
        sign_map = self.sign_map.to(self.qweight.device)
        bit_map = torch.bitwise_right_shift(
            torch.unsqueeze(sign_map, 1).expand(-1, 32, -1),
            self.wf_sign.unsqueeze(-1),
        )
        bit_map = torch.bitwise_and(bit_map, 1).to(torch.uint32)
        bit_map = bit_map.reshape(self.infeatures, self.outfeatures)
        bit_map = bit_map.t()
        weight = intweight * (self.scales.T)
        weight = weight.to(self.wf.device)
        bit_map = bit_map.to(self.wf.device)
        weight = torch.where(bit_map == 1, -weight, weight).to(torch.bfloat16)
        self.weight = weight
        return weight

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        x_dtype = x.dtype
        out = torch.zeros((x.shape[0], self.outfeatures), device=x.device, dtype=torch.float32)
        if self.ready:
            self.autogptq_cuda.vecquant4matmul(
                x.float(), #  fp8 (batch, infeatures)
                self.qweight, # int32 (infeatures // 32 * self.bits, outfeatures)
                self.sign_map, # int32 (infeatures // 32, outfeatures)
                out, #  fp16 (batch, outfeatures)
                self.scales.float(), # fp32 (1, outfeatures)
            )
        else:
            weight = self.unpack()
            weight = weight.to(x.device)
            x = x.to(torch.bfloat16)
            out = x @ weight.T
        out = out.to(x_dtype)
        out = out.reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out


__all__ = ["QuantLinear"]
