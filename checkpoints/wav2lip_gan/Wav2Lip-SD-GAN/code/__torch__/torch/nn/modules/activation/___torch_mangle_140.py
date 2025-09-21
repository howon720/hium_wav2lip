class ReLU(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.activation.___torch_mangle_140.ReLU,
    argument_1: Tensor) -> Tensor:
    return torch.relu(argument_1)
