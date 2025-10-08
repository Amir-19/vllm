# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.envs as envs
import vllm.plugins
from vllm.compilation.fix_functionalization import FixFunctionalizationPass
from vllm.compilation.fusion import RMSNormQuantFusionPass
from vllm.compilation.fx_utils import find_auto_fn, find_auto_fn_maybe, is_func
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.compilation.sequence_parallelism import SequenceParallelismPass
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.config import (CompilationConfig, CompilationLevel, DeviceConfig, ModelConfig,
                         PassConfig, VllmConfig)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (init_distributed_environment,
                                             initialize_model_parallel)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape, QuantKey, ScaleDesc)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, cutlass_fp8_supported, maybe_create_device_identity)
from vllm.platforms import current_platform
from vllm.utils import update_environment_variables

from ..utils import multi_gpu_test, override_cutlass_fp8_supported
from .backend import TestBackend


FP8_DTYPE = current_platform.fp8_dtype()


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


class TestBlockFP8Model(torch.nn.Module):
    """
    Test model that directly replicates the RowParallelLinear pattern:
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)
        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
    
    This captures the apply_w8a8_block_fp8_linear -> all_reduce pattern
    from DeepseekV2's o_proj.
    """

    def __init__(self,
                 static: bool,
                 cuda_force_torch: bool,
                 hidden_size=256,
                 intermediate_size=512,
                 vllm_config: VllmConfig = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cuda_force_torch = cuda_force_torch
        self.norm = [RMSNorm(hidden_size, eps) for _ in range(3)]
        self.wscale = [torch.rand(1, dtype=torch.float32) for _ in range(2)]
        group_shape = GroupShape.PER_TENSOR if static else GroupShape.PER_TOKEN
        quant_scale = ScaleDesc(torch.float32, static, group_shape)
        self.key = QuantKey(dtype=FP8_DTYPE, scale=quant_scale, symmetric=True)
        if static:
            self.scale = [torch.rand(1, dtype=torch.float32) for _ in range(2)]
        else:
            self.scale = [None for _ in range(2)]
        self.w = [
            torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE).t()
            for _ in range(2)
        ]

        with override_cutlass_fp8_supported(not cuda_force_torch):
            self.fp8_linear = Fp8LinearOp(
                act_quant_static=static,
                act_quant_group_shape=group_shape,
            )

    def forward(self, x):
        """
        Forward that directly implements RowParallelLinear's forward logic:
        
        1. output_parallel = self.quant_method.apply(self, input_parallel, bias_)
           -> For block FP8, this calls apply_fp8_block_linear
           -> Which internally calls torch.ops.vllm.apply_w8a8_block_fp8_linear
        
        2. if self.reduce_results and self.tp_size > 1:
               output = tensor_model_parallel_all_reduce(output_parallel)
        """
        # This is the exact pattern from RowParallelLinear.forward (line 1356)
        y = self.fp8_linear.apply(x,
                                  self.w[0],
                                  self.wscale[0],
                                  input_scale=self.scale[0])
        
        # This is the exact pattern from RowParallelLinear.forward (lines 1358-1359)
        output = tensor_model_parallel_all_reduce(y)
        return output

    def ops_in_model_before(self):
        pass

    def ops_in_model_after(self):
        pass

    def ops_in_model(self):
        pass


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("test_model_cls", [TestBlockFP8Model])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enable_fusion", [True, False])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"],
                    reason="Only test on CUDA")
def test_sequence_parallelism_pass(test_model_cls: type[torch.nn.Module],
                                   batch_size: int, seq_len: int,
                                   hidden_size: int, dtype: torch.dtype,
                                   enable_fusion: bool):
    # Skip fusion for TestBlockFP8Model
    if test_model_cls == TestBlockFP8Model and enable_fusion:
        pytest.skip("TestBlockFP8Model doesn't support fusion")
    
    # Use larger hidden_size for block FP8 (needs to work with block_size=128)
    if test_model_cls == TestBlockFP8Model:
        hidden_size = 256
    
    num_processes = 2

    def run_torch_spawn(fn, nprocs):
        # need to use torch.mp.spawn otherwise will have problems with
        # torch.distributed and cuda
        torch.multiprocessing.spawn(fn,
                                    args=(num_processes, test_model_cls,
                                          batch_size, seq_len, hidden_size,
                                          dtype, enable_fusion),
                                    nprocs=nprocs)

    run_torch_spawn(sequence_parallelism_pass_on_test_model, num_processes)


def sequence_parallelism_pass_on_test_model(
        local_rank: int, world_size: int,
        test_model_cls: type[torch.nn.Module], batch_size: int, seq_len: int,
        hidden_size: int, dtype: torch.dtype, enable_fusion: bool):
    current_platform.seed_everything(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables({
        'RANK': str(local_rank),
        'LOCAL_RANK': str(local_rank),
        'WORLD_SIZE': str(world_size),
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '12345',
    })

    # initialize distributed
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # configure vllm config for SequenceParallelismPass
    vllm_config = VllmConfig()
    vllm_config.compilation_config = CompilationConfig(pass_config=PassConfig(
        enable_sequence_parallelism=True,
        enable_fusion=enable_fusion,
        enable_noop=True))  # NoOp needed for fusion
    vllm_config.device_config = DeviceConfig(device=torch.device("cuda"))

    # this is a fake model name to construct the model config
    # in the vllm_config, it's not really used.
    model_name = "nm-testing/TinyLlama-1.1B-Chat-v1.0-FP8-e2e"
    vllm_config.model_config = ModelConfig(model=model_name,
                                           trust_remote_code=True,
                                           dtype=dtype,
                                           seed=42)

    
    vllm_config.quant_config = Fp8Config(
        weight_block_size=[128, 128],
        activation_scheme="dynamic",
    )

    # TODO(asamani): init passes here

    model = test_model_cls(hidden_size,
                           hidden_size * 2,
                           vllm_config=vllm_config)

    hidden_states = torch.randn((batch_size * seq_len, hidden_size),
                                dtype=dtype)
    
    # TODO(asamani): apply passes here


    # TODO(asamani): check pass results here
