class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.models.conv.___torch_mangle_96.Conv2d
  __annotations__["1"] = __torch__.models.conv.___torch_mangle_101.Conv2d
  __annotations__["2"] = __torch__.models.conv.___torch_mangle_106.Conv2d
  __annotations__["3"] = __torch__.models.conv.___torch_mangle_111.Conv2d
  __annotations__["4"] = __torch__.models.conv.___torch_mangle_116.Conv2d
  __annotations__["5"] = __torch__.models.conv.___torch_mangle_121.Conv2d
  __annotations__["6"] = __torch__.models.conv.___torch_mangle_126.Conv2d
  __annotations__["7"] = __torch__.models.conv.___torch_mangle_131.Conv2d
  __annotations__["8"] = __torch__.models.conv.___torch_mangle_136.Conv2d
  __annotations__["9"] = __torch__.models.conv.___torch_mangle_141.Conv2d
  __annotations__["10"] = __torch__.models.conv.___torch_mangle_146.Conv2d
  __annotations__["11"] = __torch__.models.conv.___torch_mangle_151.Conv2d
  __annotations__["12"] = __torch__.models.conv.___torch_mangle_156.Conv2d
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_157.Sequential,
    audio_sequences: Tensor) -> Tensor:
    _12 = getattr(self, "12")
    _11 = getattr(self, "11")
    _10 = getattr(self, "10")
    _9 = getattr(self, "9")
    _8 = getattr(self, "8")
    _7 = getattr(self, "7")
    _6 = getattr(self, "6")
    _5 = getattr(self, "5")
    _4 = getattr(self, "4")
    _3 = getattr(self, "3")
    _2 = getattr(self, "2")
    _1 = getattr(self, "1")
    _0 = getattr(self, "0")
    _13 = (_1).forward((_0).forward(audio_sequences, ), )
    _14 = (_4).forward((_3).forward((_2).forward(_13, ), ), )
    _15 = (_7).forward((_6).forward((_5).forward(_14, ), ), )
    _16 = (_10).forward((_9).forward((_8).forward(_15, ), ), )
    _17 = (_12).forward((_11).forward(_16, ), )
    return _17
