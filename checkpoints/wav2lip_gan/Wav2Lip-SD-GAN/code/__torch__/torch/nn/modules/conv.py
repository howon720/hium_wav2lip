class Conv2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.conv.Conv2d,
    face_sequences: Tensor) -> Tensor:
    bias = self.bias
    weight = self.weight
    input = torch._convolution(face_sequences, weight, bias, [1, 1], [3, 3], [1, 1], False, [0, 0], 1, False, False, True, True)
    return input
class ConvTranspose2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.conv.ConvTranspose2d,
    input: Tensor) -> Tensor:
    bias = self.bias
    weight = self.weight
    input0 = torch._convolution(input, weight, bias, [1, 1], [0, 0], [1, 1], True, [0, 0], 1, False, False, True, True)
    return input0
