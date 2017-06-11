# This script adapts the APIs of the different attacks to
# a unique one.
import ctypes_utils
import numpy as np
from ctypes import cdll
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

# Attacks
from dyer.Trace import Trace
from dyer.Packet import Packet
from dyer.classifiers.LiberatoreClassifier import LiberatoreClassifier
from dyer.classifiers.WrightClassifier import WrightClassifier
#from dyer.classifiers.HerrmannClassifier import HerrmannClassifier
from dyer.classifiers.PanchenkoClassifier import PanchenkoClassifier
from dyer.classifiers.VNGPlusPlusClassifier import VNGPlusPlusClassifier
#from dyer.TimeClassifier import TimeClassifier
#from dyer.TimeClassifier import TimeClassifier
#from dyer.BandwidthClassifier import BandwidthClassifier
#from dyer.VNGClassifier import VNGClassifier
#from dyer.JaccardClassifier import JaccardClassifier
from hayes.RF_fextract import TOTAL_FEATURES as kfp_extract_features
from hayes.kFP import RF_closedworld as kfp_classify_closed
from hayes.kFP import RF_openworld_adapted as kfp_classify_open
from hayes.kFP import RF_openworld_features as kfp_open_features
from wang.fextractor import extract as knn_extract_features

WANG_LIB = 'attacks/wang/flearner.so'
NAN = np.nan
ISNAN = np.isnan


class Attack(object):
    """Common interface for all the attacks.
    """

    def evaluate_attack(self, train_fv, train_labels, test_fv, test_labels,
                         world='closed'):
        """Returns a dictionary containing the attack's results
        (e.g., error, TP/FP (if world=='open')).
        """
        res = {}
        pred = self.classify(train_fv, train_labels, test_fv, test_labels)
        
        # Accuracy.
        n_test = len(test_fv)
        acc = 0.0
        for p, t in zip(pred, test_labels):
            if p == t:
                acc += 1
        acc /= n_test
        res['error'] = 1 - acc
        # Need to have them as list of int
        res['true_labels'] = [int(x) for x in test_labels]
        res['predicted_labels'] = [int(x) for x in pred]

        return res

    def extract_features(self, packet_seq, labels=None):
        """Accepts a list of packet_sequences, returns
        a list of feature vectors.

        The input argument `labels' is required for compatibility
        with Dyer's code.
        """
        pass

    def classify(self, train_fv, train_labels, test_fv, test_labels):
        """Returns a list of label predictions for test_fv,
        training the classifier on (train_fv, train_labels).
        """
        pass


# DYER code abstraction.
class DyerAttack(Attack):
    """Common interface to abstract attacks implemented
    by Dyer.
    """

    def __init__(self, attack):
        """Configures the methods with respect to attack.

        Sets the class' method "trace_to_features" to the
        attack's static method traceToInstance.
        The method will be called to extract features.
        Sets the method to call for classification.
        """
        self.trace_to_instance = attack.traceToInstance
        self.call_classify = attack.classify

    def classify(self, train_fv, train_labels, test_fv, test_labels):
        """Returns the accuracy of the attack trained on trainining
        feature vectors train_fv, and performed on test_fv.
        """
        runid = self.__class__.__name__
        print "Converting vectors to Dyer's format."
        train_in = self._feature_vecs_to_instances(train_fv, train_labels)
        test_in = self._feature_vecs_to_instances(test_fv, test_labels)

        print "Performing classification using Weka."
        _, debug = self.call_classify(runid, train_in, test_in)
        # Unpack list of (true_label, pred_label)
        _, pred = zip(*debug)
        pred = [int(x) for x in pred]

        return pred

    def extract_features(self, packet_seqs, labels):
        """Extract feature vectors from packet sequences.
        
        Converts the list of packet sequences (packet_seqs)
        into a list of Trace, and extracts features in Dyer format
        ("instances").
        """
        instances = []
        feature_names = []
        for p, l in zip(packet_seqs, labels):
            t = self._packet_sequence_to_trace(p, l)
            f = self.trace_to_instance(t)
            instances.append(f)
            feature_names += f.keys()

        feature_names = list(set(feature_names))
        feature_names.remove('class')
        feature_vecs = self._instances_to_feature_vecs(instances,
                                                       feature_names)

        return feature_vecs

    def _feature_vecs_to_instances(self, feature_vecs, labels):
        """Converts a list of feature vectors to a list of
        instances.

        Note: as explained in _instances_to_feature_vecs(),
        there is no need to check for missing values, because
        _instances_to_feature_vecs() sets any missing value
        to 0. This reflects the behaviour of arffWriter.py.
        """
        instances = []
        d = len(feature_vecs[0])

        for x, l in zip(feature_vecs, labels):
            f = {'class': str(l)}
            for i in range(d):
                name = str(i)       # The feature name is simply its position.
                f[name] = x[i]
            instances.append(f)

        return instances

    def _instances_to_feature_vecs(self, instances, names):
        """Converts a list of instances to a list of feature
        vectors.

        Note: Dyer's arffWriter.py sets any missing value to 0
        before running a classifier. This function adopts the
        same behaviour.
        """
        feature_vecs = []
        for f in instances:
            x = []
            for k in sorted(names):
                if not k in f:
                    x.append(0)
                else:
                    x.append(f[k])
            feature_vecs.append(x)

        return feature_vecs

    def _packet_sequence_to_trace(self, packet_seq, label):
        """Converts a packet sequence into a Trace object.
        """
        trace = Trace(label)
        for time, size in packet_seq:
            if size >= 0:
                direction = Packet.UP
            else:
                direction = Packet.DOWN
            packet = Packet(direction, time, abs(size))
            trace.addPacket(packet)

        return trace

# Defining all Dyer's attacks as classes.
class LiberatoreAttack(DyerAttack):
    def __init__(self):
        super(self.__class__, self).__init__(LiberatoreClassifier)

class WrightAttack(DyerAttack):
    def __init__(self):
        super(self.__class__, self).__init__(WrightClassifier)

class HerrmannAttack(DyerAttack):
    def __init__(self):
        super(self.__class__, self).__init__(HerrmannClassifier)

class PanchenkoAttack(DyerAttack):
    def __init__(self):
        super(self.__class__, self).__init__(PanchenkoClassifier)

class VNGPlusPlusAttack(DyerAttack):
    def __init__(self):
        super(self.__class__, self).__init__(VNGPlusPlusClassifier)


# HAYES code abstraction.
class KFPAttack(Attack):
    def classify(self, train_fv, train_labels, test_fv, test_labels,
                 mode='closed', k=3):
        """Returns the accuracy of the attack trained on trainining
        feature vectors train_fv, and performed on test_fv.
        If mode is 'closed', closed world classification is used
        (i.e., only random forest).
        If mode is 'open', random forest is used to generate leaves,
        and k-NN is used for prediction.
        If mode is 'open', k is the number of neighbours for k-NN.
        """
        if mode == 'closed':
            print 'Using KFP in Closed World setting'
            _, pred = kfp_classify_closed(train_fv, train_labels, test_fv,
                                          test_labels)
        elif mode == 'open':
            print 'Using KFP in Open World setting'
            _, pred = kfp_classify_open(train_fv, train_labels, test_fv,
                                        test_labels, k)
        else:
            raise Exception('Mode {} not known'.format(mode))
        
        return pred
    
    def extract_features(self, packet_seqs, labels=None):
        """Extract feature vectors from packet sequences.
        """
        feature_vecs = [kfp_extract_features(p) for p in packet_seqs]

        return feature_vecs

    def _features_to_open_world(self, train_fv, train_labels):
        """Extracts features (leaves) using RF from training objects.
        """
        test_fv = train_fv[0]       # Needed for how the API works
        new_fv, _ = kfp_open_features(train_fv, train_labels, test_fv)

        return new_fv.tolist()


# WANG code abstraction.
class KNNAttack(Attack):
    
    def __init__(self):
        self.libknn = cdll.LoadLibrary(WANG_LIB)

    def classify(self, train_fv, train_labels, test_fv, test_labels, k=2):
        """Returns the accuracy of the attack trained on trainining
        feature vectors train_fv, and performed on test_fv.
        """
        n_train = len(train_fv)
        n_test = len(test_fv)
        n_labels = len(set(test_labels))

        print 'Learning weights.'
        weights = self._recommend_weights(train_fv, train_labels)
        
        # Prepare return array.
        predictions = [-1]*n_test

        # Convert to ctypes.
        train_fv_c = ctypes_utils.list_list_to_c_float_list_list(train_fv)
        test_fv_c = ctypes_utils.list_list_to_c_float_list_list(test_fv)
        train_labels_c = ctypes_utils.list_to_c_int_list(train_labels)
        predictions_c = ctypes_utils.list_to_c_int_list(predictions)
        weights_c = ctypes_utils.list_to_c_float_list(weights)

        print 'Making prediction.'
        self.libknn.knn_predict(train_fv_c, train_labels_c, test_fv_c,
                                ctypes_utils.c_int(n_train),
                                ctypes_utils.c_int(n_test),
                                ctypes_utils.c_int(n_labels),
                                ctypes_utils.c_int(k),
                                weights_c, predictions_c)

        # Get back results.
        predictions = ctypes_utils.c_list_to_list(predictions_c)

        return predictions

    def extract_features(self, packet_seqs, labels=None):
        """Extract feature vectors from packet sequences.
        """
        feature_vecs = []
        for p in packet_seqs:
            # Separate into packet times and sizes
            times, sizes = zip(*p)
            # Extract the features for one packet sequence
            x = knn_extract_features(times, sizes)
            # Replace missing values with -1
            x = self._impute_missing(x)
            feature_vecs.append(x)

        return feature_vecs

    def _init_weights(self, d):
        """Returns a list of initialised weights.
        Weights are initialised randomly using numpy library.
        You can set the general seed of the library if you
        want reproducible results.

        Params
        d : int
            Length of the vector of weights.
        """
        return [np.random.rand() + 0.5 for x in range(d)]


    def _recommend_weights(self, feature_vecs, labels, n_rounds=5):
        """Recommends weights for the features.
        """
        n = len(feature_vecs)
        d = len(feature_vecs[0])

        # Initialise weights.
        weights = self._init_weights(d)

        # Convert to ctypes.
        n_c = ctypes_utils.c_int(n)
        feature_vecs_c = ctypes_utils.list_list_to_c_float_list_list(feature_vecs)
        labels_c = ctypes_utils.list_to_c_int_list(labels)
        weights_c = ctypes_utils.list_to_c_float_list(weights)

        for r in range(n_rounds):
            print 'Round {}'.format(r)
            start = n/n_rounds * r
            end = n/n_rounds * (r+1)
            # The result is put into weights_c.
            self.libknn.recommend_weight(feature_vecs_c, labels_c, n_c,
                                         weights_c,
                                         ctypes_utils.c_int(start),
                                         ctypes_utils.c_int(end))
        # Convert from ctypes.
        weights = ctypes_utils.c_list_to_list(weights_c)

        return weights
    
    def _impute_missing(self, x):
        """Accepts a list of features containing 'X' in
        place of missing values. Consistently with the code
        by Cai et al, replaces 'X' with -1.
        """
        for i in range(len(x)):
            if x[i] == 'X':
                x[i] = -1
        
        return x


class CUMULAttack(Attack):

    def classify(self, train_fv, train_labels, test_fv, test_labels, seed=0,
                 nfolds=5):
        """Returns the accuracy of the attack trained on trainining
        feature vectors train_fv, and performed on test_fv.
        Following Panchenko's et al., the code first determines the
        best parameters (C, gamma) using grid search (using the same
        grid as the paper).
        It then uses SVM to make predictions.
        """
        print 'Scaling features'
        train_fv = preprocessing.scale(train_fv)
        test_fv = preprocessing.scale(test_fv)

        print 'Looking for the best parameters using {}-folds CV'.format(nfolds)
        # NOTE: these are the parameters that worked best on the
        # WCN+ dataset among those I tried
        C_range = np.logspace(11, 17, 5, base=2.0)
        gamma_range = np.logspace(-3, 3, 2, base=2.0)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=nfolds, test_size=0.2,
                                        random_state=seed)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(train_fv, train_labels)
	
        print 'Best parameters using grid search: {}'.format(grid.best_params_)
        C = grid.best_params_['C']
        gamma = grid.best_params_['gamma']
	
        print 'Fitting SVM'
        svm = SVC(C=C, gamma=gamma)
        svm.fit(train_fv, train_labels)

        print 'Predicting'
        pred = svm.predict(test_fv)
        e = sum(pred != np.array(test_labels)) / float(len(test_labels))
        print 'Error: {}'.format(e)
        
        return pred
    
    def extract_features(self, packet_seqs, labels=None, n=100):
        """Extract feature vectors from packet sequences.
        n is the number of points to interpolate.
        """
        feature_vecs = [self._extract_features(p, n) for p in packet_seqs]

        return feature_vecs

    def _extract_features(self, packet_seq, n):
        """Extracts a feature vector from a packet sequence.
        n is the number of points to interpolate.
        """
        fv = []

        # Basic features
        in_size = 0
        out_size = 0
        in_count = 0
        out_count = 0

        # Init cumulative features
        _, size = packet_seq[0]
        c = [size]
        a = [abs(size)]

        if len(packet_seq) > 1:
            for _, size in packet_seq[1:]:
                if size > 0:
                    in_size += size
                    in_count += 1
                elif size < 0:
                    out_size += abs(size)
                    out_count += 1
                else:
                    # Skip if size 0
                    continue

                c.append(c[-1] + size)
                a.append(a[-1] + abs(size))

        # Interpolate cumulative features
        cumul = np.interp(np.linspace(a[0], a[-1], n), a, c)

        fv = [in_size, out_size, in_count, out_count] + list(cumul)

        return fv


# All attacks.
ATTACKS = {'ll': LiberatoreAttack,
           'wright': WrightAttack,
           #'herrmann': HerrmannAttack,
           #'time': TimeAttack,
           #'bdw': BandwidthAttack,
           'panchenko': PanchenkoAttack,
           #'vng': VNGAttack,
           'vng++': VNGPlusPlusAttack,
           #'jacc': JaccardAttack,
           'knn': KNNAttack,
           'kfp': KFPAttack,
           'CUMUL': CUMULAttack,
          }
