class Conv2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv_block : __torch__.torch.nn.modules.container.Sequential
  act : __torch__.torch.nn.modules.activation.ReLU
  def forward(self: __torch__.models.conv.Conv2d,
    face_sequences: Tensor) -> Tensor:
    act = self.act
    conv_block = self.conv_block
    _0 = (conv_block).forward(face_sequences, )
    return (act).forward(_0, )
class Conv2dTranspose(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv_block : __torch__.torch.nn.modules.container.___torch_mangle_165.Sequential
  act : __torch__.torch.nn.modules.activation.___torch_mangle_166.ReLU
  def forward(self: __torch__.models.conv.Conv2dTranspose,
    input: Tensor) -> Tensor:
    act = self.act
    conv_block = self.conv_block
    _1 = (act).forward((conv_block).forward(input, ), )
    return _1
