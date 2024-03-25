# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Qwen2Moe model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Qwen2MoEConfig(PretrainedConfig):
    r"""
        This is the configuration class to store the configuration of a [`Qwen2MoeModel`]. It is used to instantiate a
        Qwen2 MoE model according to the specified arguments, defining the model architecture.

        Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
        documentation from [`PretrainedConfig`] for more information.


        Args:
            vocab_size (`int`, *optional*, defaults to 151936):
                Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
                `inputs_ids` passed when calling [`Qwen2MoeModel`]
            hidden_size (`int`, *optional*, defaults to 4096):
                Dimension of the hidden representations.
            intermediate_size (`int`, *optional*, defaults to 22016):
                Dimension of the MLP representations.
            num_hidden_layers (`int`, *optional*, defaults to 32):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (`int`, *optional*, defaults to 32):
                Number of attention heads for each attention layer in the Transformer encoder.
            num_key_value_heads (`int`, *optional*, defaults to 32):
                This is the number of key_value heads that should be used to implement Grouped Query Attention. If
                `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
                `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
                converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
                by meanpooling all the original heads within that group. For more details checkout [this
                paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
            hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
                The non-linear activation function (function or string) in the decoder.
            max_position_embeddings (`int`, *optional*, defaults to 32768):
                The maximum sequence length that this model might ever be used with.
            initializer_range (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            rms_norm_eps (`float`, *optional*, defaults to 1e-06):
                The epsilon used by the rms normalization layers.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not the model should return the last key/values attentions (not used by all models). Only
                relevant if `config.is_decoder=True`.
            tie_word_embeddings (`bool`, *optional*, defaults to `False`):
                Whether the model's input and output word embeddings should be tied.
            rope_theta (`float`, *optional*, defaults to 10000.0):
                The base period of the RoPE embeddings.
            use_sliding_window (`bool`, *optional*, defaults to `False`):
                Whether to use sliding window attention.
            sliding_window (`int`, *optional*, defaults to 4096):
                Sliding window attention (SWA) window size. If not specified, will default to `4096`.
            max_window_layers (`int`, *optional*, defaults to 28):
                The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
            attention_dropout (`float`, *optional*, defaults to 0.0):
                The dropout ratio for the attention probabilities.
            num_experts_per_tok (`int`, *optional*, defaults to 2):
                The number of experts to root per-token, can be also interpreted as the `top-p` routing
                parameter
            num_local_experts (`int`, *optional*, defaults to 8):
                Number of experts per Sparse MLP layer.
            output_router_logits (`bool`, *optional*, defaults to `False`):
                Whether or not the router logits should be returned by the model. Enabeling this will also
                allow the model to output the auxiliary loss. See [here]() for more details
            router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
                The aux loss factor for the total loss.

    ```python
    # >>> from modeling_qwen2_moe import Qwen2MoE, MixtralConfig
    # >>> from modeling_qwen2_moe import

    >>> # Initializing a Mixtral 7B style configuration
    >>> configuration = Qwen2MoEConfig()

    >>> # Initializing a model from the Mixtral 7B style configuration
    >>> model = Qwen2MoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1e6,
        sliding_window=None,
        max_window_layer=28,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.max_window_layer = max_window_layer

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )