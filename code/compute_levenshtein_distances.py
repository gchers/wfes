import os
import sys
import dill
import argparse
import numpy as np
import multiprocessing
#from leven import levenshtein
from editdistance import eval as levenshtein
from utils.experiment_utils import log
from utils.data_utils import load_dataset
from scipy.spatial.distance import pdist, squareform

def compute_levenshtein(args):
    i, x1, x2 = args

    return i, levenshtein(x1, x2)

def pairs(X):
    """Return iterator over pairs of strings in X.
    """
    m = len(X)
    k = 0
    for i in xrange(m-1):
        for j in xrange(i+1, m):
            yield k, X[i], X[j]
            k += 1

def pairwise_levenshtein_distances(X):
    """Returns the pairwise Levenshtein distance of the strings
    in list X.
    """
    M = len(X)
    pool = multiprocessing.Pool()#processes=8) #, maxtasksperchild=10)

    r = pool.map_async(compute_levenshtein, pairs(X))
    #results = []
    #for k, (x1, x2) in enumerate(pairs(X)):
    #    r = pool.apply_async(compute_levenshtein, args=(k, x1, x2))
    #    results.append(r)

    del X

    tot = ((M * (M - 1)) / 2)

    percentage = 0
    while not r.ready():
        left = r._number_left * r._chunksize
        new_percentage = int((1 - left/float(tot)) * 100)
        if new_percentage > percentage:
            percentage = new_percentage
            print("Progress: {}\r".format(percentage))
            sys.stdout.flush()

    results = r.get()

    print
    print("To square form")
    # Reorder and put to square form
    dm = [None] * tot
    for k, dist in results:
        dm[k] = dist
    #dm = [None] * len(results)
    #for r in results:
    #    k, dist = r.get()
    #    dm[k] = dist
    for d in dm:
        if d is None:
            print("None found in distances")

    return squareform(dm)

def encode_sizes(X):
    """Extracts packet sizes, encodes outgoing sizes with "1",
    incoming with "0", and returns a string for each packet
    sequence.
    """
    # Only keep packet size
    sizes = []
    for x in X:
        ps = [s for _, s in x]
        # Encode -1 => '0', 1 => '1'
        ps = ['0' if s == -1 else '1' for s in ps]
        # Convert into strings
        sizes.append(''.join(ps))

    return sizes

def run(traces, outfname):
    X, Y, W, _, _ = load_dataset(traces)

    sizes = encode_sizes(X)

    log('Computing pairwise distances')
    D = pairwise_levenshtein_distances(sizes)
    log('Computing subtractions')
    
    log('Storing distances into {}'.format(outfname))
    
    data = {'webpage-id': W,
            'label': np.array(Y),
            'pairdist': D,
           }

    with open(outfname, 'wb') as f:
        dill.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute distances')
    parser.add_argument('--traces', type=str, help='Traces directory.',
                        required=True)
    parser.add_argument('--out', type=str, help='Distance file (.distances).',
                        required=True)
    args = parser.parse_args()

    run(args.traces, args.out)
