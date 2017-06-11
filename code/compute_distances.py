import os
import dill
import argparse
import numpy as np
import pandas as pd
from utils.experiment_utils import log
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from utils.data_utils import load_features, load_dataset
from utils.experiment_utils import train_test_closed_world

DISTANCE = 'euclidean'

def knn_attack_distance(u, v):
    """Returns the distance proposed in the k-NN attack by Wang
    et al. 2014.

    NOTE: this is not a proper distance metric.
    """
    if len(u) != len(v):
        raise Exception("knn_attack_distance: u, v should have the same length")
    d = 0.0
    for i in range(len(u)):
        if u[i] != -1 and v[i] != -1:
            d += abs(u[i] - v[i])

    return d

def pairwise_distances(X, knn_attack_dist=False):
    """Returns the pairwise distances of the objects in
    X.
    """
    # Obtain distance matrix (each row contains the distances
    # from one object).
    if not knn_attack_dist:
        print 'Using {} distance'.format(DISTANCE)
        return squareform(pdist(X, metric=DISTANCE))
    else:
        print 'Using k-NN distance'
        return squareform(pdist(X, metric=knn_attack_distance))

def packet_sequences_only(X):
    P = []
    # Find maximum packet sequence length
    maximum = 0
    for p in X:
        if len(p) > maximum:
            maximum = len(p)
    # Pad to maximum length, in place
    log('Padding each sequence to {}'.format(len(p)))
    for i in xrange(len(X)):
        # Flatten and pad
        X[i] = [x for p in X[i] for x in p] + [0]*(2*(maximum - len(X[i])))

    return np.array(X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute distances')
    parser.add_argument('--features', type=str, help='Features directory.',
                        required=True)
    parser.add_argument('--sequences', action='store_true',
                        help='Compute on packet sequences.', required=False,
                        default=False)
    parser.add_argument('--out', type=str, help='Distance file (.distances).',
                        required=True)
    args = parser.parse_args()

    if not args.sequences:
        X, Y, W, _, _ = load_features(args.features)
    else:
        X, Y, W, _, _ = load_dataset(args.features)
        X = packet_sequences_only(X)

    log('Computing pairwise distances')
    D = pairwise_distances(X)
    log('Computing subtractions')
    
    log('Storing distances into {}'.format(args.out))
    
    data = {'webpage-id': W,
            'label': np.array(Y),
            'pairdist': D,
           }

    with open(args.out, 'wb') as f:
        dill.dump(data, f)
