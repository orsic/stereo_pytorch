import numpy as np
import sys

if __name__ == '__main__':
    filename = sys.argv[1]
    splits = {}
    with open(filename, 'r') as file:
        for line in file.readlines():
            line = line.rstrip()
            split, prec, perc = tuple(line.split(' '))
            split = split[:-1]
            hit, miss = tuple(prec.split('/'))
            hm = (int(hit), int(miss))
            if split not in splits:
                splits[split] = [hm]
            else:
                splits[split].append(hm)
    for split in splits:
        prec = np.array(splits[split])
        hits = np.sum(prec[:,0])
        total = np.sum(prec[:, 1])
        print('{} {}'.format(split, 100 * hits / total))