class Conv2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv_block : __torch__.torch.nn.modules.container.___torch_mangle_104.Sequential
  act : __torch__.torch.nn.modules.activation.___torch_mangle_105.ReLU
  def forward(self: __torch__.models.conv.___torch_mangle_106.Conv2d,
    argument_1: Tensor) -> Tensor:
    act = self.act
    conv_block = self.conv_block
    input = torch.add_((conv_block).forward(argument_1, ), argument_1)
    return (act).forward(input, )
