def revert(path, output_path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f]

    with open(output_path, 'w') as f:
        for line in lines:
            f.write(revert_line(line) + '\n')


def revert_line(line):
    result = []
    for token in line.split():
        if token.startswith('<sp>'):
            token = token[4:-5]
            result += token.split('$@$')
        else:
            result.append(token)

    return ' '.join(result)


if __name__ == '__main__':
    revert('output/merged-small.txt', 'output/reverted-small.txt')
