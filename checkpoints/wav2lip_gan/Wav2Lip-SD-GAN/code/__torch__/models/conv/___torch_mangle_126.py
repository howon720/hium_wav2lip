class Conv2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv_block : __torch__.torch.nn.modules.container.___torch_mangle_124.Sequential
  act : __torch__.torch.nn.modules.activation.___torch_mangle_125.ReLU
  def forward(self: __torch__.models.conv.___torch_mangle_126.Conv2d,
    argument_1: Tensor) -> Tensor:
    act = self.act
    conv_block = self.conv_block
    _0 = (act).forward((conv_block).forward(argument_1, ), )
    return _0
