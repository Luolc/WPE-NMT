""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_final, memory_bank = self.encoder(src, lengths)
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state


class WPEModel(nn.Module):
    """
        Core trainable object in OpenNMT. Implements a trainable interface
        for a simple, generic encoder + decoder model.

        Args:
          encoder (:obj:`EncoderBase`): an encoder object
          decoder (:obj:`RNNDecoderBase`): a decoder object
        """

    def __init__(self, encoder, decoder, pair_size=3):
        super(WPEModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.pair_size = pair_size

        self.deconv = nn.ConvTranspose1d(decoder.hidden_size, decoder.hidden_size, pair_size)

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[(tgt_pair_len * pair_size) x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        enc_final, memory_bank = self.encoder(src, lengths)
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)

        assert tgt.size(0) % 3 == 0
        if tgt.size(0) > 3:
            tgt = tgt[:-self.pair_size]  # exclude last pair from inputs

        emb = self.decoder.embeddings(tgt)

        # (l*pair_size x batch x dim) -> (l x pair_size x batch x dim)
        emb = emb.view(-1, self.pair_size, emb.size(1), emb.size(2))
        emb = emb.sum(dim=1)

        decoder_outputs, dec_state, attns = \
            self.decoder(emb, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths)

        # deconvolution: (len*batch x hidden x pair_size)
        deconved = self.deconv(decoder_outputs.view(-1, decoder_outputs.size(2), 1))
        deconved = deconved.view(decoder_outputs.size(0),
                                 decoder_outputs.size(1),
                                 decoder_outputs.size(2),
                                 self.pair_size)
        deconved = deconved.permute(0, 3, 1, 2).contiguous().view(
            -1, decoder_outputs.size(1), decoder_outputs.size(2))

        return deconved, attns, dec_state
