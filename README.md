# Debiasing the Denoising U-Net: Training a LoRA to Mitigate Bias in Stable Diffusion

See the `diffusers` and `peft` repositories for proof on how the LoRA is applied to the cross-attention layers across the U-Net.

In the `diffusers` repository in the file `src/diffusers/loaders/unet.py`, within the class `UNet2DConditionLoadersMixin`, see the functions `fuse_lora()` and `_fuse_lora_apply()`.

- `self.apply(partial(self._fuse_lora_apply, adapter_names=adapter_names))` is calling the `apply()` method with a partially applied function `partial(self._fuse_lora_apply, adapter_names=adapter_names)`. This means that the `_fuse_lora_apply` method will be called on each submodule of the current module, with the adapter_names argument passed to it, meaning that it will be applied to each cross-attention head in the U-Net.

- The `_fuse_lora_apply` method calls `module.merge(**merge_kwargs)` to merge the LoRA layers into the module, and this happend across all the modules in the U-Net.

```python
def fuse_lora(self, lora_scale=1.0, safe_fusing=False, adapter_names=None):
    self.lora_scale = lora_scale
    self._safe_fusing = safe_fusing
    self.apply(partial(self._fuse_lora_apply, adapter_names=adapter_names))

def _fuse_lora_apply(self, module, adapter_names=None):
    if not USE_PEFT_BACKEND:
        ...
    else:
        from peft.tuners.tuners_utils import BaseTunerLayer

        merge_kwargs = {"safe_merge": self._safe_fusing}

        if isinstance(module, BaseTunerLayer):
            if self.lora_scale != 1.0:
                module.scale_layer(self.lora_scale)

            # For BC with prevous PEFT versions, we need to check the signature
            # of the `merge` method to see if it supports the `adapter_names` argument.
            supported_merge_kwargs = list(inspect.signature(module.merge).parameters)
            if "adapter_names" in supported_merge_kwargs:
                merge_kwargs["adapter_names"] = adapter_names
            elif "adapter_names" not in supported_merge_kwargs and adapter_names is not None:
                raise ValueError(
                    "The `adapter_names` argument is not supported with your PEFT version. Please upgrade"
                    " to the latest version of PEFT. `pip install -U peft`"
                )

            module.merge(**merge_kwargs)
```

In the `peft` repository, in the `src/peft/tuners/lora/lora.py` script, see the `merge()` function in the `Linear` class.

- For each active adapter (`active_adapter`) in `adapter_names`, which would just be a list with 1 element in this case containing the name of the LoRA, the method checks if the adapter's weights (`self.lora_A` and `self.lora_B`) exist.

- If the adapter's weights exist, the method retrieves the base layer (`base_layer`) using `self.get_base_layer()`. The base layer is the original linear layer that LoRA is being applied to, which in this case would be the cross-attention head.

- `safe_merge` is set to False, so the method directly merges the LoRA weights with the base layer's weights. The delta weight (`delta_weight`) is obtained using `self.get_delta_weight(active_adapter)`, which is calculated as `transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]`.

- Finally, the merged adapter is appended to the self.merged_adapters list to keep track of which adapters have been merged.

- In the context of cross-attention heads, the Linear class would be used to apply LoRA to the query and value projection matrices. The merge method would be called to merge the LoRA weights with the base weights of these projection matrices, effectively modifying the behavior of the cross-attention heads to obtain some of the "debiasing" behavior we attempt to inject into it.

```python
class Linear(nn.Module, LoraLayer):
    ...
    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        # ...
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # ...
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data = base_layer.weight.data + delta_weight
                    else:
                        # handle dora
                        # ...
                self.merged_adapters.append(active_adapter)
```

The LoRA is thus added onto the QKV projections. 
