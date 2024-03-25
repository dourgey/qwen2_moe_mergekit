# qwen2_moe_mergekit
根据Qwen2（Qwen1.5）模型生成qwen2 MoE模型的工具

# 使用方法：
参见`convert_kit.py`的`main`函数

# BUGs
- [ ] 模型保存后，使用`AutoModel`进行加载时，如果环境中未安装`flash-attn`会报错，这时需要注释掉`src.modeling_qwen2_moe.py`的55~59行，即：
```
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
```
- [ ] 在某些版本的`transformers`库下加载保存的模型时，可能会出错，建议升级到`requirements.txt`的版本，其他版本未完整测试
