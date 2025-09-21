class ReLU(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.activation.ReLU,
    argument_1: Tensor) -> Tensor:
    return torch.relu(argument_1)
class Sigmoid(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.activation.Sigmoid,
    argument_1: Tensor) -> Tensor:
    return torch.sigmoid(argument_1)
