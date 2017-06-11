import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def nn_bound_cv(D, Y, n_folds=5):
    """Uses CV to compute the NN error bound.
    """
    nn = KNeighborsClassifier(n_neighbors=1, metric='precomputed',
                              algorithm='brute', n_jobs=-1)
    # NN error using CV
    cv_scores = 1 - cross_val_score(nn, D, Y, cv=n_folds)
    # Compute the bound for each score
    L = float(len(set(Y)))
    bounds = []
    for e in cv_scores:
        e_bounded = min(e, (L-1)/L)
        bound = (L-1)/L * (1 - math.sqrt(1 - L/(L-1)*e_bounded))
        bounds.append(bound)

    return bounds

def knn(D, Y, K=None):
    """Compute k-NN bounds on dataset D with labels Y.
    Returns the results in a nested dictionary.
    """
    if not K:
        K = range(1, len(D))

    # Init results (counting errors)
    res = {'knn-vote': {'LOO': {},
                        'resubstitution': {},
                        'rejection': {}
                       },
           'nn-bound': 0.0
          }
    for k in K:
        for method in ['LOO', 'resubstitution']:
            res['knn-vote'][method][k] = 0.0
        res['knn-vote']['rejection'][k] = 0.0

    # Compute NN error bound using CV
    try:
        res['nn-bound-5-cv'] = nn_bound_cv(D, Y, 5)
    except:
        res['nn-bound-5-cv'] = []
    try:
        res['nn-bound-10-cv'] = nn_bound_cv(D, Y, 10)
    except:
        res['nn-bound-10-cv'] = []

    # Compute other bounds
    for i in range(len(D)):
        d = D[i,:]
        y = Y[i]

        # Exclude the i-th object
        d_i = np.delete(d, i)
        Y_i = np.delete(Y, i)

        # Sort by increasing distances
        sorted_idx = np.argsort(d_i)
        d_sort = d_i[sorted_idx]
        Y_sort = Y_i[sorted_idx]

        # Distance from i-th object for each class
        max_k = -1           # Maximum k for Volume k-NN
        d_class = {}
        for c in set(Y):
            d_class[c] = d_sort[Y_sort==c]
            if len(d_class[c]) < max_k or max_k == -1:
                max_k = len(d_class[c])

        for k in K:
            # Vote k-NN (LOO)
            count = np.bincount(Y_sort[:k])
            y_hat = count.argmax()
            if y_hat != y:
                res['knn-vote']['LOO'][k] += 1
            # Vote k-NN (rejection)
            k1 = int(math.ceil(k/2.0))
            # Accept only if qualifying majority k1 + 1 is achieved
            # otherwise, do not count the prediction
            if count.max() >= k1 + 1:
                if y_hat != y:
                    res['knn-vote']['rejection'][k] += 1
            # Vote k-NN (resubstitution)
            y_hat = np.bincount(np.concatenate(([y],Y_sort[:k-1]))).argmax()
            if y_hat != y:
                res['knn-vote']['resubstitution'][k] += 1

    # Average errors
    for k in K:
        for method in ['LOO', 'resubstitution']:
            res['knn-vote'][method][k] /= len(D)
        res['knn-vote']['rejection'][k] /= len(D)

    # Compute the Bayes bound based on NN error
    L = float(len(set(Y)))
    nn_error = res['knn-vote']['LOO'][1]
    nn_error_bounded = min(nn_error, (L-1)/L)
    nn_bound = (L-1)/L * (1 - math.sqrt(1 - L/(L-1)*nn_error_bounded))
    res['nn-bound'] = nn_bound
    
    return res
