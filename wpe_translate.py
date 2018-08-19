from __future__ import print_function
import argparse
import codecs
import os
import math

import torch

from itertools import count
from onmt.utils.misc import tile

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.opts as opts


def translate(opt):
    out_file = codecs.open(opt.output, 'w+', 'utf-8')

    if opt.gpu > -1:
        torch.cuda.set_device(opt.gpu)

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, model, model_opt = \
        onmt.model_builder.load_test_model(opt, dummy_opt.__dict__)

    data = inputters.build_dataset(fields,
                                   'text',
                                   src_path=opt.src,
                                   src_data_iter=None,
                                   tgt_path=opt.tgt,
                                   tgt_data_iter=None,
                                   src_dir=opt.src_dir,
                                   sample_rate='16000',
                                   window_size=.02,
                                   window_stride=.01,
                                   window='hamming',
                                   use_filter_pred=False)

    device = torch.device('cuda' if opt.gpu > -1 else 'cpu')

    batch_size = 1

    data_iter = inputters.OrderedIterator(
        dataset=data, device=device,
        batch_size=batch_size, train=False, sort=False,
        sort_within_batch=True, shuffle=False)

    pair_size = model_opt.wpe_pair_size
    s_id = fields["tgt"].vocab.stoi['<s>']
    ss_id = fields["tgt"].vocab.stoi['<sgo>']

    for i, batch in enumerate(data_iter):
        tgt = torch.LongTensor([s_id] * batch_size + [ss_id] * (2 * batch_size)).view(
            pair_size, batch_size).unsqueeze(2).to(device)
        dec_state = None
        src = inputters.make_features(batch, 'src', 'text')
        _, src_lengths = batch.src

        result = None

        for _ in range(opt.max_length):
            outputs, _, dec_state = model(src, tgt, src_lengths, dec_state)
            scores = model.generator(outputs.view(-1, outputs.size(2)))
            indices = scores.argmax(dim=1)
            tgt = indices.view(pair_size, batch_size, 1)  # (pair_size x batch x feat)

            if result is None:
                result = indices.view(pair_size, batch_size)
            else:
                result = torch.cat([result, indices.view(pair_size, batch_size)], 0)

        result = result.transpose(0, 1).tolist()
        for sent in result:
            sent = [fields["tgt"].vocab.itos[_] for _ in sent]
            sent = ' '.join(sent)
            out_file.write(sent + '\n')

        print('Translated {} batches'.format(i))

    out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    translate(opt)
