from collections import defaultdict
import copy
import re


def process(src, dst, n_merges, verbose=False):
    out_stream = open(dst, 'w+')

    sentences = get_sentences(src.format('train'))
    sentences += get_sentences(src.format('test'))

    stats, indices = get_pair_statistics(sentences)
    big_stats = copy.deepcopy(stats)
    # threshold is inspired by Zipfian assumption, but should only affect speed
    threshold = max(stats.values()) / 10

    most_frequent = max(stats, key=lambda x: (stats[x], x))

    for i in range(n_merges):
        if stats:
            most_frequent = max(stats, key=lambda x: (stats[x], x))

        # we probably missed the best pair because of pruning; go back to full statistics
        if not stats or (i and stats[most_frequent] < threshold):
            prune_stats(stats, big_stats, threshold)
            stats = copy.deepcopy(big_stats)
            most_frequent = max(stats, key=lambda x: (stats[x], x))
            # threshold is inspired by Zipfian assumption, but should only affect speed
            threshold = stats[most_frequent] * i / (i + 10000.0)
            prune_stats(stats, big_stats, threshold)

        if verbose:
            print('pair {0}: {1} {2} -> "{1} {2}" (frequency {3})'.format(
                i, most_frequent[0], most_frequent[1], stats[most_frequent]))

        out_stream.write('{} {}\n'.format(*most_frequent))
        changes = replace_pair(most_frequent, sentences, indices)
        update_pair_statistics(most_frequent, changes, stats, indices)
        stats[most_frequent] = 0
        if not i % 100:
            prune_stats(stats, big_stats, threshold)

    out_stream.close()


def get_sentences(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    sentences = [tuple(l.split()) for l in lines]
    return sentences


def get_pair_statistics(sentences):
    """Count frequecy of all token pairs, and create index."""
    stats = defaultdict(int)
    indices = defaultdict(lambda: defaultdict(int))

    for i, tokens in enumerate(sentences):
        if len(tokens) < 2:
            continue
        prev_token = tokens[0]
        for token in tokens[1:]:
            stats[prev_token, token] += 1
            indices[prev_token, token][i] += 1
            prev_token = token

    return stats, indices


def prune_stats(stats, big_stats, threshold):
    """Prune statistics dict for efficiency of max()

    The frequency of a symbol pair never increases, so pruning is generally safe
    (until the most frequent pair is less frequent than a pair we previously pruned)
    big_stats keeps full statistics for when we need to access pruned items
    """
    for item, freq in list(stats.items()):
        if freq < threshold:
            del stats[item]
            if freq < 0:
                big_stats[item] += freq
            else:
                big_stats[item] = freq


def replace_pair(pair, sentences, indices):
    """Replace all occurrences of a token pair ('ABC', 'XYZ') with a new symbol 'ABC XYZ'"""
    first, second = pair
    pair_str = ' '.join(pair)
    pair_str = pair_str.replace('\\', '\\\\')
    changes = []
    pattern = re.compile(re.escape(first + '\t' + second))
    for j, freq in indices[pair].items():
        if freq < 1:
            continue
        tokens = sentences[j]
        new_snetence = '\t'.join(tokens)
        new_snetence = pattern.sub(pair_str, new_snetence)
        new_snetence = tuple(new_snetence.split('\t'))

        sentences[j] = new_snetence
        changes.append((j, new_snetence, tokens))

    return changes


def update_pair_statistics(pair, changed, stats, indices):
    """Minimally update the indices and frequency of symbol pairs

    if we merge a pair of symbols, only pairs that overlap with occurrences
    of this pair are affected, and need to be updated.
    """
    stats[pair] = 0
    indices[pair] = defaultdict(int)
    first, second = pair
    new_pair = first + ' ' + second
    for j, new_sentence, old_sentence in changed:
        # find all instances of pair, and update frequency/indices around it
        i = 0
        while True:
            # find first symbol
            try:
                i = old_sentence.index(first, i)
            except ValueError:
                break
            # if first symbol is followed by second symbol,
            # we've found an occurrence of pair (old_word[i:i+2])
            if i < len(old_sentence) - 1 and old_sentence[i + 1] == second:
                # assuming a symbol sequence "A B C", if "B C" is merged,
                # reduce the frequency of "A B"
                if i:
                    prev = old_sentence[i - 1:i + 1]
                    stats[prev] -= 1
                    indices[prev][j] -= 1
                if i < len(old_sentence) - 2:
                    # assuming a symbol sequence "A B C B",
                    # if "B C" is merged, reduce the frequency of "C B".
                    # however, skip this if the sequence is A B C B C,
                    # because the frequency of "C B" will be reduced by the previous code block
                    if old_sentence[i + 2] != first or i >= len(old_sentence) - 3 \
                            or old_sentence[i + 3] != second:
                        nex = old_sentence[i + 1:i + 3]
                        stats[nex] -= 1
                        indices[nex][j] -= 1
                i += 2
            else:
                i += 1

        i = 0
        while True:
            try:
                # find new pair
                i = new_sentence.index(new_pair, i)
            except ValueError:
                break
            # assuming a symbol sequence "A BC D", if "B C" is merged,
            # increase the frequency of "A BC"
            if i:
                prev = new_sentence[i - 1:i + 1]
                stats[prev] += 1
                indices[prev][j] += 1
            # assuming a symbol sequence "A BC B", if "B C" is merged,
            # increase the frequency of "BC B"
            # however, if the sequence is A BC BC, skip this step
            # because the count of "BC BC" will be incremented by the previous code block
            if i < len(new_sentence) - 1 and new_sentence[i + 1] != new_pair:
                nex = new_sentence[i:i + 2]
                stats[nex] += 1
                indices[nex][j] += 1
            i += 1


if __name__ == '__main__':
    process('data/zh_en/tgt-{}.small.txt', 'data/zh_en/small.wpe.txt', 50000, verbose=True)
