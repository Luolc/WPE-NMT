from collections import defaultdict


def apply(data_path, output_path, wpe_path, n_merges, pair_size=3):
    wpes = get_wpes(wpe_path, n_merges, pair_size)

    with open(data_path, 'r') as f:
        lines = [l.strip() for l in f]

    with open(output_path, 'w') as f:
        for i, line in enumerate(lines):
            for token in line.split():
                pairs = wpes[token]
                for raw, replacement in pairs:
                    line = line.replace(raw, replacement)

            f.write(line + '\n')

            if i % 2000 == 0:
                print('Merged {} sentences'.format(i))


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

    for key, pairs in result.items():
        result[key] = [(' '.join(_), '<sp>' + '$@$'.join(_) + '</sp>') for _ in result[key]]

    return result


if __name__ == '__main__':
    apply('data/zh_en/tgt-train.txt',
          'data/zh_en/tgt-train.merged.txt',
          'data/zh_en/tgt-train.wpe.txt',
          30000)
