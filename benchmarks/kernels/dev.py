import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    apply_w8a8_block_fp8_linear,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_BLOCK_FP8_SUPPORTED,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton as vllm_triton

assert current_platform.is_cuda(), (
    "Only support benchmarking w8a8 block fp8 kernel on CUDA device."
)

from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
  


# DeepSeek-V3 weight shapes
DEEPSEEK_V3_SHAPES = [
    (512 + 64, 7168),
    (2112, 7168),
    ((128 + 64) * 128, 7168),
    (128 * (128 + 128), 512),
    (7168, 16384),
    (7168, 18432),
    (18432 * 2, 7168),
    (24576, 1536),
    (12288, 7168),
    (4096, 7168),
    (7168, 2048),
]

def build_w8a8_block_fp8_runner(M, N, K, block_size, device):
    factor_for_scale = 1e-2

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    
    # Create random FP8 tensors
    A_ref = (torch.rand(M, K, dtype=torch.bfloat16, device=device) - 0.5) * 2 * fp8_max

    B_ref = (torch.rand(N, K, dtype=torch.bfloat16, device=device) - 0.5) * 2 * fp8_max
    B = B_ref.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    # Create scales
    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    Bs = (
        torch.rand(n_tiles, k_tiles, dtype=torch.float32, device=device)
        * factor_for_scale
    )
    # float8_e8m0fnu
    # SM90 CUTLASS requires row-major format for scales
    if current_platform.is_device_capability(90):
        Bs = Bs.T.contiguous()

    input_2d = A_ref.view(-1, A_ref.shape[-1])
    output_shape = [*A_ref.shape[:-1], A_ref.shape[0]]
    output_dtype = A_ref.dtype

    # deepgemm style: input -> A_ref
    # input_2d = input.view(-1, input.shape[-1])
    # output_shape = [*input.shape[:-1], weight.shape[0]]

    # q_input, x_scale = per_token_group_quant_fp8(
    #     input_2d,
    #     block_size[1],
    #     column_major_scales=True,
    # )
    #output = torch.empty((q_input.shape[0], weight.shape[0]),
    #  dtype=torch.bfloat16,
    #  device=q_input.device)
    # fp8_gemm_nt((q_input, x_scale), (weight, weight_scale), output)

    # cutlass style: 
    # q_input, x_scale = per_token_group_quant_fp8(input_2d,
    #                                                  block_size[1],
    #                                                  column_major_scales=True)
    # output = w8a8_blockscale_func(q_input, weight, x_scale, weight_scale,
    #                               block_size, input.dtype)
    q_input, x_scale = per_token_group_quant_fp8(input_2d,
                                                     block_size[1],
                                                     column_major_scales=True)
    out = torch.empty((q_input.shape[0], B.shape[0]),
                          dtype=torch.bfloat16,
                          device=q_input.device)
    masked_m = torch.full((1,), M, dtype=torch.int32, device=q_input.device)
    ab_dtype = "float8_e4m3fn"
    sf_dtype = "float32"
    sf_vec_size = 32
    c_dtype = "bfloat16"
    def run():
        output_cutlass =  apply_w8a8_block_fp8_linear(
                A_ref, B, block_size, Bs, cutlass_block_fp8_supported=True
            )
        
        output_cute = grouped_gemm_nt_masked(
            (q_input, x_scale), 
            (B, Bs), # w1.permute(1, 2, 0)??
            out,
            masked_m,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype=c_dtype,
            sf_vec_size=sf_vec_size,
            )
        
        return output_cutlass, output_cute
    return run

def bench(batch_size, N, K, block_size=(128, 128)):
    M = batch_size
    device = "cuda"

    run_w8a8_cutlass = build_w8a8_block_fp8_runner(
            M, N, K, block_size, device
        )
    out = run_w8a8_cutlass()
    print(out)

if __name__ == "__main__":
    block_size = (128, 128)
    batch_size = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    for N, K in DEEPSEEK_V3_SHAPES:
        print(f"M = {batch_size[4]}, N={N}, K={K}")
        bench(
            batch_size=batch_size[4],
            N=N,
            K=K,
            block_size=block_size,)
        break