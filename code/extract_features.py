import os
import argparse
import numpy as np
from utils.data_utils import load_dataset
from attacks.attacks import ATTACKS
from attacks.hayes.kFP import RF_openworld_features

def log(s):
    print '[*] {}'.format(s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract feature vectors')
    parser.add_argument('--traces', type=str, help='Traces.',
                        required=True)
    parser.add_argument('--attack', type=str, help='Attack.',
                        required=True, choices=ATTACKS.keys())
    parser.add_argument('--type', type=str, help='knn/kfp',
                        required=False)
    parser.add_argument('--out', type=str, help='Output directory.',
                        required=True)
    args = parser.parse_args()

    
    X, Y, W, Npages, Nloads = load_dataset(args.traces)

    log('Considering features of attack {}'.format(args.attack))
    attack = ATTACKS[args.attack]()

    log('Extracting features')
    X_f = attack.extract_features(X, Y)

    log('Lenght of a feature vector: {}'.format(len(X_f)))
    
    if args.type:
        if args.type == 'knn':
            log('Features specific to k-NN attack')
            weights = attack._recommend_weights(X_f, Y)
            log('Applying weights')
            X_new = []
            for xf in X_f:
                X_new.append([x*w for x,w in zip(xf,weights)])
            X_f = X_new
        elif args.type == 'kfp':
            log('Features specific to k-FP attack')
            X_f = attack._features_to_open_world(X_f, Y)

    if not os.path.isdir(args.out):
        log('Creating directory {}'.format(args.out))
        os.makedirs(args.out)

    log('Storing features into {}'.format(args.out))
    for x, w in zip(X_f, W):
        fname = os.path.join(args.out, w) + '.features'
        np.savetxt(fname, x, delimiter=',')
