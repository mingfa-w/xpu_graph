RemoveLayerNormCast pattern
```mermaid
graph TD
subgraph DST
    layernorm_[torch.ops.aten.native_layer_norm.default] --> getitem_[operator.getitem]
end
subgraph SRC
    pre_cast[torch.ops.aten._to_copy.default] --> layernorm[torch.ops.aten.native_layer_norm.default]
    layernorm --> getitem[operator.getitem]
    getitem --> post_cast[torch.ops.aten._to_copy.default]
end
```