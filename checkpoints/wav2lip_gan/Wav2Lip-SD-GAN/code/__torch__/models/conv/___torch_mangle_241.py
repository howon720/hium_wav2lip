class Conv2dTranspose(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv_block : __torch__.torch.nn.modules.container.___torch_mangle_239.Sequential
  act : __torch__.torch.nn.modules.activation.___torch_mangle_240.ReLU
  def forward(self: __torch__.models.conv.___torch_mangle_241.Conv2dTranspose,
    input: Tensor) -> Tensor:
    act = self.act
    conv_block = self.conv_block
    _0 = (act).forward((conv_block).forward(input, ), )
    return _0
