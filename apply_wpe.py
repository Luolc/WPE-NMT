from collections import defaultdict
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data/zh_en/tgt-train.txt')
    parser.add_argument('--output_path', type=str, default='data/zh_en/tgt-train.merged.txt')
    parser.add_argument('--wpe_path', type=str, default='data/zh_en/tgt-train.wpe.txt')
    parser.add_argument('--n_merges', type=int, default=30000)
    parser.add_argument('--pair_size', type=int, default=3)
    parser.add_argument('--segment_size', type=int, default=-1)
    parser.add_argument('--index', type=int, default=0)

    return parser.parse_args()


def apply(data_path, output_path, wpe_path, n_merges, pair_size=3, segment_size=-1, index=0):
    wpes = get_wpes(wpe_path, n_merges, pair_size)

    with open(data_path, 'r') as f:
        lines = [l.strip() for l in f]

    if segment_size != -1:
        lines = lines[index * segment_size:index * segment_size + segment_size]

    with open(output_path, 'w') as f:
        for line_index, line in enumerate(lines):
            tokens = tuple(line.split())
            pairs = [wpe for token in tokens for wpe in wpes[token]]

            line = ' '.join(['<word>{}</word>'.format(_) for _ in line.split()])

            for raw, replacement in pairs:
                line = line.replace(raw, replacement)

            replaced = line.replace('<word>', '').replace('</word>', '')
            f.write(replaced + '\n')

            if line_index % 2000 == 0:
                print('Merged {} sentences'.format(line_index))


def get_wpes(wpe_path, n_merges, pair_size):
    with open(wpe_path, 'r') as f:
        lines = [l.strip() for l in f]
    lines = [l.split() for l in lines]
    lines = [tuple(l) for l in lines if len(l) <= pair_size]
    lines = lines[:n_merges]

    result = defaultdict(list)
    for pair in lines:
        result[pair[0]].append(pair)

    for _, pairs in result.items():
        pairs.sort(key=lambda x: len(x), reverse=True)

    _wpe_to_text = lambda _wpe: ' '.join(['<word>{}</word>'.format(_) for _ in _wpe])
    _wpe_to_replacement = lambda _wpe: '<sp>{}</sp>'.format('$@$'.join(_wpe))

    for key, pairs in result.items():
        result[key] = [(_wpe_to_text(wpe), _wpe_to_replacement(wpe)) for wpe in pairs]

    return result


if __name__ == '__main__':
    args = get_args()
    apply(args.data_path,
          args.output_path,
          args.wpe_path,
          args.n_merges,
          args.pair_size,
          args.segment_size,
          args.index)
