import argparse
from attacks.attacks import ATTACKS
from utils.experiment_utils import log
from utils.data_utils import load_features
from utils.data_utils import store_results
from sklearn.model_selection import train_test_split
from utils.experiment_utils import train_test_closed_world

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an attack')
    parser.add_argument('--features', type=str, help='Features directory.',
                        required=True)
    parser.add_argument('--attack', type=str, help='Attack.',
                        required=True, choices=ATTACKS.keys())
    parser.add_argument('--train', type=float,
                        help='Percentage of training instances.', 
                        required=True)
    parser.add_argument('--test', type=float,
                        help='Percentage (or number) of test instances.',
                        required=True)
    parser.add_argument('--seed', type=int, help='PRNG seed (default: 0).',
                        required=False, default=0)
    parser.add_argument('--out', type=str, help='Output file (.json).',
                        required=True)
    args = parser.parse_args()

    
    X, Y, _, Npages, Nloads = load_features(args.features)

    log('Seed is {}'.format(args.seed))

    n = len(X)
    # Get training/test set size
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

    log('Splitting into training/test set keeping uniform labels')
    # First get the test set
    I = range(n)                        # Indexes
    Iother, Itest = train_test_split(I, test_size=test_size, stratify=Y,
                                     random_state = args.seed)
    Xtest = X[Itest,:]
    Ytest = Y[Itest]
    # Now the training set
    # If train_size + test_size < n, sample from the remaining
    # n - test_size instances the training set
    if train_size < len(Iother):
        log('Reduced training set')
        # Strain_size instances from Iother to create the training set
        Itrain, _ = train_test_split(Iother, train_size=train_size,
                                     stratify=Y[Iother], random_state=args.seed)
    else:
        Itrain = Iother
    # Training set
    Xtrain = X[Itrain,:]
    Ytrain = Y[Itrain]

    if not args.attack in ATTACKS:
        raise Exception("Attack {} not found".format(attack.args))
    log('Using attack {}'.format(args.attack))
    attack = ATTACKS[args.attack]()

    log('Running the attack')
    res = attack.evaluate_attack(Xtrain, Ytrain, Xtest, Ytest)

    print 'Results: {}'.format(res)
    log('Storing results into {}'.format(args.out))
    store_results(args.out, res)
