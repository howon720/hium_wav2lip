class Conv2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv_block : __torch__.torch.nn.modules.container.___torch_mangle_94.Sequential
  act : __torch__.torch.nn.modules.activation.___torch_mangle_95.ReLU
  def forward(self: __torch__.models.conv.___torch_mangle_96.Conv2d,
    audio_sequences: Tensor) -> Tensor:
    act = self.act
    conv_block = self.conv_block
    _0 = (conv_block).forward(audio_sequences, )
    return (act).forward(_0, )
