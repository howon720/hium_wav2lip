class Wav2Lip(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  face_encoder_blocks : __torch__.torch.nn.modules.container.ModuleList
  audio_encoder : __torch__.torch.nn.modules.container.___torch_mangle_157.Sequential
  face_decoder_blocks : __torch__.torch.nn.modules.container.___torch_mangle_253.ModuleList
  output_block : __torch__.torch.nn.modules.container.___torch_mangle_260.Sequential
  def forward(self: __torch__.models.wav2lip.Wav2Lip,
    audio_sequences: Tensor,
    face_sequences: Tensor) -> Tensor:
    output_block = self.output_block
    face_decoder_blocks = self.face_decoder_blocks
    _6 = getattr(face_decoder_blocks, "6")
    face_decoder_blocks0 = self.face_decoder_blocks
    _5 = getattr(face_decoder_blocks0, "5")
    face_decoder_blocks1 = self.face_decoder_blocks
    _4 = getattr(face_decoder_blocks1, "4")
    face_decoder_blocks2 = self.face_decoder_blocks
    _3 = getattr(face_decoder_blocks2, "3")
    face_decoder_blocks3 = self.face_decoder_blocks
    _2 = getattr(face_decoder_blocks3, "2")
    face_decoder_blocks4 = self.face_decoder_blocks
    _1 = getattr(face_decoder_blocks4, "1")
    face_decoder_blocks5 = self.face_decoder_blocks
    _0 = getattr(face_decoder_blocks5, "0")
    face_encoder_blocks = self.face_encoder_blocks
    _60 = getattr(face_encoder_blocks, "6")
    face_encoder_blocks0 = self.face_encoder_blocks
    _50 = getattr(face_encoder_blocks0, "5")
    face_encoder_blocks1 = self.face_encoder_blocks
    _40 = getattr(face_encoder_blocks1, "4")
    face_encoder_blocks2 = self.face_encoder_blocks
    _30 = getattr(face_encoder_blocks2, "3")
    face_encoder_blocks3 = self.face_encoder_blocks
    _20 = getattr(face_encoder_blocks3, "2")
    face_encoder_blocks4 = self.face_encoder_blocks
    _10 = getattr(face_encoder_blocks4, "1")
    face_encoder_blocks5 = self.face_encoder_blocks
    _00 = getattr(face_encoder_blocks5, "0")
    audio_encoder = self.audio_encoder
    _7 = (audio_encoder).forward(audio_sequences, )
    _8 = (_00).forward(face_sequences, )
    _9 = (_10).forward(_8, )
    _11 = (_20).forward(_9, )
    _12 = (_30).forward(_11, )
    _13 = (_40).forward(_12, )
    _14 = (_50).forward(_13, )
    _15 = (_60).forward(_14, )
    input = torch.cat([(_0).forward(_7, ), _15], 1)
    input0 = torch.cat([(_1).forward(input, ), _14], 1)
    input1 = torch.cat([(_2).forward(input0, ), _13], 1)
    input2 = torch.cat([(_3).forward(input1, ), _12], 1)
    input3 = torch.cat([(_4).forward(input2, ), _11], 1)
    input4 = torch.cat([(_5).forward(input3, ), _9], 1)
    input5 = torch.cat([(_6).forward(input4, ), _8], 1)
    return (output_block).forward(input5, )
