
def to_str(x, type):
    if x is None:
        return ''
    if type is float:
        # return f'{x:.4f}'
        return f'{100*x:.2f}'
    return str(x)

def print_table(rows, headers, types):
    def _print_row(row, lens):
        for i, x in enumerate(row):
            s = f'| {x}'
            s += ' ' * (lens[i] - len(x) + 1)
            print(s, end='')
        print('|')

    def _print_hor_line(lens):
        for i, x in enumerate(lens):
            print('+', end='')
            print('-' * (x + 2), end='')
        print('+')

    rows = [[to_str(x, t) for x, t in zip(row, types)] for row in rows]
    # Get len of columns
    col_lens = [len(h) for h in headers]
    for row in rows:
        for i in range(len(col_lens)):
            col_lens[i] = max(col_lens[i], len(row[i]))

    def print_row(row): _print_row(row, col_lens)
    def print_hor_line(): _print_hor_line(col_lens)

    # Print
    print_hor_line()
    print_row(headers)
    print_hor_line()
    for row in rows:
        print_row(row)
    print_hor_line()

def dump_table(rows, headers, types, file):
    file.parent.mkdir(exist_ok=True, parents=True)
    rows = [[to_str(x, t) for x, t in zip(row, types)] for row in rows]
    
    with open(file, 'w') as f:
        f.write('\t'.join(headers) + '\n')
        for row in rows:
            f.write('\t'.join(row) + '\n')

