#!/bin/sh
python3 preprocess.py \
    -train_src=data/zh_en/src-train.small.txt \
    -train_tgt=data/zh_en/tgt-train.small.txt \
    -valid_src=data/zh_en/src-valid.txt \
    -valid_tgt=data/zh_en/tgt-valid.txt \
    -save_data=data/zh_en/b.s \
    -src_vocab_size=30000 \
    -tgt_vocab_size=20000 \
    -src_words_min_frequency=3 \
    -tgt_words_min_frequency=3 \
    -src_seq_length=50 \
    -tgt_seq_length=50