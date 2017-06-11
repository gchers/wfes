import dill
import argparse
import numpy as np
from evaluate import bounds
from sklearn.utils import resample
from utils.experiment_utils import log
from utils.data_utils import store_results
from sklearn.model_selection import train_test_split
from utils.experiment_utils import train_test_closed_world

def split_training_test(D, Y, train_size, test_size):
    """Splits distances D and labels Y into training and test
    sets as explained below.

    Note: train_size + test_size <= len(D)

    The test set of size test_size is first sampled.
    Then, a training set of size train_size is sampled from the
    remaining examples.
    This allows to vary the size of the training set, keeping
    a fixed test set.

    If train_size + test_size = len(D), the function simply
    splits (D,Y) into a training and test set using all the
    examples.
    """
    n = len(D)
    # NOTE: we need to remove both rows AND columns from
    # the distance matrix.
    log('Splitting into training/test set keeping uniform labels')
    # First get the test set
    I = range(n)                        # Indexes
    Iother, Itest = train_test_split(I, test_size=test_size, stratify=Y,
                                     random_state = args.seed)
    Ytest = Y[Itest]
    # Need to do this in two steps
    Dtest = D[Itest,:]
    Dtest = D[:,Itest]
    # Now the training set
    if train_size < len(Iother):
        log('Reduced training set')
        # Now sample train_size instances from Iother to create the training set
        Itrain, _ = train_test_split(Iother, train_size=train_size,
                                     stratify=Y[Iother], random_state=args.seed)
    else:
        Itrain = Iother
    log('Training set has size {}'.format(len(Itrain)))
    Ytrain = Y[Itrain]
    # Need to do this in two steps
    Dtrain = D[Itrain,:]
    Dtrain = Dtrain[:,Itrain]

    return Dtrain, Ytrain, Dtest, Ytest

def one_vs_all_setting(D, Y, target, seed=0):
    """Prepares the dataset (D, Y) to a one-vs-all setting
    with the specified target.
    Returns a new D, Y.
    Note that target label is set to 0, open world label
    to 1.
    """
    # Relabel target to 0, open to 1
    if np.any(Y==-1):
        raise Exception("Labels should not have value -1")
    Y[Y==target] = -1
    Y[Y!=-1] = 1
    Y[Y==-1] = 0
    # Keep a uniform number of examples with respect
    # to their label (target, open)
    np.random.seed(seed)
    target_idx = np.where(Y==0)[0]
    open_idx = np.where(Y==1)[0]
    # NOTE: I assume there are always more examples from
    # the open world than those from target
    idx = np.random.choice(open_idx, len(target_idx))
    keep_idx = np.concatenate((idx, target_idx))
    # Need to do the following in two steps
    D = D[keep_idx,:]
    D = D[:,keep_idx]
    Y = Y[keep_idx]

    return D, Y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute bounds')
    parser.add_argument('--distances', type=str, help='.distances file.',
                        required=True)
    parser.add_argument('--train', type=float,
                        help='Percentage (or number) of training instances.',
                        required=True)
    parser.add_argument('--test', type=float,
                        help='Percentage (or number) of test instances.',
                        required=True)
    parser.add_argument('--seed', type=int, help='PRNG seed (default: 0).',
                        required=False, default=0)
    parser.add_argument('--bootstrap', help='Use bootstrap.',
                        action='store_true', default=False)
    parser.add_argument('--target', help='Target page for 1 vs All setting.',
                        required=False, type=int)
    parser.add_argument('--out', type=str, help='Results file (.json).',
                        required=True)
    args = parser.parse_args()

    log('Loading distances from {}'.format(args.distances))
    with open(args.distances, 'rb') as f:
        data = dill.load(f)

    D = data['pairdist']
    Y = np.array(data['label'])

    if args.target is not None:
        log('One-vs-all setting using {} as target'.format(args.target))
        log('Reducing the dataset for one-vs-all')
        D, Y = one_vs_all_setting(D, Y, args.target)

    log('Seed is {}'.format(args.seed))

    # (Maybe) apply bootstrap
    if args.bootstrap:
        log('Bootstrapping')
        D, Y = resample(D, Y, replace=True, n_samples=len(D),
                        random_state=args.seed)

    # Parse training/test set size
    n = len(D)
    if args.train > 1:
        train_size = int(args.train)
    else:
        train_size = int(args.train*n)
    if args.test > 1:
        test_size = int(args.test)
    else:
        test_size = int(args.test*n)

    log('Training set size: {}. Test set size: {}.'.format(train_size,
                                                             test_size))
    if train_size + test_size != n:
        log('Note: {} instances will be excluded'.format(n - train_size - test_size))

    # Split into training/test set, only keep training
    if train_size == len(D):
        log('Using all data as training set')
        Dtrain = D
        Ytrain = Y
    else:
        Dtrain, Ytrain, _, _ = split_training_test(D, Y, train_size, test_size)

    log('Computing k-NN related bounds on the training set')
    res = bounds.knn(Dtrain, Ytrain)

    log('Storing results into {}'.format(args.out))
    store_results(args.out, res)
