class ModuleList(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.torch.nn.modules.container.___torch_mangle_0.Sequential
  __annotations__["1"] = __torch__.torch.nn.modules.container.___torch_mangle_16.Sequential
  __annotations__["2"] = __torch__.torch.nn.modules.container.___torch_mangle_37.Sequential
  __annotations__["3"] = __torch__.torch.nn.modules.container.___torch_mangle_53.Sequential
  __annotations__["4"] = __torch__.torch.nn.modules.container.___torch_mangle_69.Sequential
  __annotations__["5"] = __torch__.torch.nn.modules.container.___torch_mangle_80.Sequential
  __annotations__["6"] = __torch__.torch.nn.modules.container.___torch_mangle_91.Sequential
class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.torch.nn.modules.conv.Conv2d
  __annotations__["1"] = __torch__.torch.nn.modules.batchnorm.BatchNorm2d
  def forward(self: __torch__.torch.nn.modules.container.Sequential,
    face_sequences: Tensor) -> Tensor:
    _1 = getattr(self, "1")
    _0 = getattr(self, "0")
    _2 = (_1).forward((_0).forward(face_sequences, ), )
    return _2
