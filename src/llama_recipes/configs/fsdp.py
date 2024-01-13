# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class fsdp_config:
    """
    FSDP(Fully Sharded Data Parallel)
    1. bf16 : bfloat16의 준말, 16비트 부동 소수점 형식
    - 32비트 부동 소수점 형식보다는 정확도가 떨어지지만, 메모리 요구 사항이 적으므로 모델 학습에 유용
    2. fp16 : half-precision의 준말로, 16비트 부동 소수점 형식
    - 메모리를 적게 사용하지만, 16비트의 정밀도가 낮아서 모델의 정확도가 떨어질 수 있음
    - 모델 학습시에는 fp32를 사용하고, 추론시에는 fp16을 사용하는 것이 일반적
    - https://jaeyung1001.tistory.com/entry/bf16-fp16-fp32%EC%9D%98-%EC%B0%A8%EC%9D%B4%EC%A0%90
    """
    mixed_precision: bool=True
    use_fp16: bool=False
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT  # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    fsdp_activation_checkpointing: bool=True
    fsdp_cpu_offload: bool=False
    pure_bf16: bool = False
    optimizer: str= "AdamW"