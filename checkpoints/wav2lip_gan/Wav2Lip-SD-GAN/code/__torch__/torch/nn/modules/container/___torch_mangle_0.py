class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.models.conv.Conv2d
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_0.Sequential,
    face_sequences: Tensor) -> Tensor:
    _0 = getattr(self, "0")
    return (_0).forward(face_sequences, )
