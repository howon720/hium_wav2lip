class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.models.conv.Conv2dTranspose
  __annotations__["1"] = __torch__.models.conv.___torch_mangle_171.Conv2d
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_172.Sequential,
    input: Tensor) -> Tensor:
    _1 = getattr(self, "1")
    _0 = getattr(self, "0")
    _2 = (_1).forward((_0).forward(input, ), )
    return _2
