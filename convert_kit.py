from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, Qwen2Config
# from src.modeling_qwen2_moe import Qwen2MoEForCausalLM
from src.configuration_qwen2_moe import Qwen2MoEConfig
import re
import shutil
from src.modeling_qwen2_moe import Qwen2MoEForCausalLM


# qwen_model = AutoModelForCausalLM.from_pretrained(r"D:\AI\Qwen1.5-0.5B")
# qwen_tokenizer = AutoTokenizer.from_pretrained(r"D:\AI\Qwen1.5-0.5B")
#
# qwen_moe_config = AutoConfig.from_pretrained(r"test/", trust_remote_code=True)
# qwen_moe_model = Qwen2MoEForCausalLM(qwen_moe_config)

def convert_qwen2_config_to_qwen2_moe_config(
        qwen2_config: Qwen2Config,
        num_experts_per_tok: int = 2,
        num_local_experts: int = 8,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
    ) -> Qwen2MoEConfig:
    """
    Convert a Qwen2Config object to a Qwen2MoEConfig object.

    Args:
        qwen2_config (Qwen2Config): The input Qwen2Config object.
        num_experts_per_tok (int): Number of experts per token.
        num_local_experts (int): Number of local experts.
        output_router_logits (bool): Whether to output router logits.
        router_aux_loss_coef (float): Coefficient for router auxiliary loss.

    Returns:
        Qwen2MoEConfig: The converted Qwen2MoEConfig object.
    """
    qwen2_moe_config = Qwen2MoEConfig(**qwen2_config.to_dict())
    qwen2_moe_config.architectures = ["Qwen2MoEForCausalLM"]
    del qwen2_moe_config._name_or_path

    qwen2_moe_config.num_experts_per_tok = num_experts_per_tok
    qwen2_moe_config.num_local_experts = num_local_experts
    qwen2_moe_config.output_router_logits = output_router_logits
    qwen2_moe_config.router_aux_loss_coef = router_aux_loss_coef
    return qwen2_moe_config


def get_parametered_layers_list(model, num_hidden_layers):
    """
    Given a model and the number of hidden layers, this function returns two lists:
    one containing the parameterized layers of the model, grouped by hidden layer,
    and the other containing the remaining parameterized layers of the model.

    :param model: A PyTorch model object.
    :param num_hidden_layers: An integer representing the number of hidden layers in the model.
    :return: A tuple containing two lists - one for the parameterized layers of the model, grouped by hidden layer,
             and the other for the remaining parameterized layers of the model.
    """
    layers = []
    for name, module in model.named_parameters():
        layers.append((name, module))
    hidden_layers = [[] for _ in range(num_hidden_layers)]

    other_layers = []

    for layer in layers:
        if re.search("layers\.\d+", layer[0]):
            hidden_layers[int(layer[0].split("layers.")[1].split(".")[0])].append(layer)
        else:
            other_layers.append(layer)

    return hidden_layers, other_layers


def copy_weight(model, moe_model):
    """
    Copy weights from model to moe_model for matching layers.
    Args:
        model: The source model to copy weights from.
        moe_model: The destination model to copy weights to.
    """
    assert model.config.num_hidden_layers == moe_model.config.num_hidden_layers, "num_hidden_layers is not matched."
    num_hidden_layers = model.config.num_hidden_layers

    model_hidden_layers, model_other_layers = get_parametered_layers_list(model, model.config.num_hidden_layers)
    moe_hidden_layers, moe_other_layers = get_parametered_layers_list(moe_model, moe_model.config.num_hidden_layers)

    assert set([other_layer[0] for other_layer in model_other_layers]) == set([other_layer[0] for other_layer in moe_other_layers]), "Other layer is not matched."

    model_other_layers = {layer[0]: layer[1] for layer in model_other_layers}
    moe_other_layers = {layer[0]: layer[1] for layer in moe_other_layers}

    for other_layer in model_other_layers:
        moe_other_layers[other_layer].data = model_other_layers[other_layer].data

    for i in range(num_hidden_layers):
        model_layer = model_hidden_layers[i]
        moe_layer = moe_hidden_layers[i]

        attn_weights = {x[0]: x[1] for x in model_layer if "mlp" not in x[0]}
        mlp_weights = {x[0].split("mlp.")[1]: x[1] for x in model_layer if "mlp" in x[0]}

        for weight_name, weight in moe_layer:
            if weight_name in attn_weights:
                weight.data.copy_(attn_weights[weight_name].data)

            else:
                if "block_sparse_moe.gate" in weight_name:
                    continue
                weight.data.copy_(mlp_weights[re.split("experts\.\d+\.", weight_name)[1]].data)


# copy_weight(qwen_model, qwen_moe_model)
#
def count_parameters(module):
    """
    统计给定nn.Module的参数量。

    参数:
    module (nn.Module): 要统计参数的模型或层。

    返回:
    int: 参数的总数。
    """
    total_params = 0
    for param in module.parameters():
        if param.requires_grad:
            total_params += param.numel()  # numel() 返回参数元素的总数
    return total_params


class Qwen2ModelForCausalLM:
    pass


def convert_causal_lm_to_moe_causal_lm(
        model_name_or_path: str,
        output_dir: str,
        num_experts_per_tok: int = 2,
        num_local_experts: int = 8,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001):

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = model.config

    qwen2_moe_config = convert_qwen2_config_to_qwen2_moe_config(config, num_experts_per_tok, num_local_experts, output_router_logits, router_aux_loss_coef)
    moe_model = Qwen2MoEForCausalLM(qwen2_moe_config)

    print("MoE model config: ", qwen2_moe_config.to_json_string())

    print("Origin model parameters: ", count_parameters(model))
    print("Moe model parameters: ", count_parameters(moe_model))
    print("Moe model Parameters ratio: ", count_parameters(moe_model) / count_parameters(model))

    print("copying weights...")

    copy_weight(model, moe_model)

    moe_model.config.auto_map = {
        "AutoConfig": "configuration_qwen2_moe.Qwen2MoEConfig",
        "AutoModelForCausalLM": "modeling_qwen2_moe.Qwen2MoEForCausalLM"
    }

    moe_model.half()

    moe_model.save_pretrained(output_dir)
    # qwen2_moe_config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # copy src.modeling_qwen2_moe.py / src.configuration_qwen2_moe.py to output_dir
    shutil.copy("src/configuration_qwen2_moe.py", output_dir)
    shutil.copy("src/modeling_qwen2_moe.py", output_dir)

    print("Done! Saved in ", output_dir)


if __name__ == '__main__':
    origin_model_path = r""
    moe_model_save_path = r""
    convert_causal_lm_to_moe_causal_lm(origin_model_path, moe_model_save_path)


