#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <string.h>
#include <sstream>
#include <time.h>
#include <stdlib.h>
#include <algorithm>
using namespace std;

//Data parameters
int FEAT_NUM = 3736; //number of features

//const int SITE_NUM = 100; //number of monitored sites
//const int INST_NUM = 60; //number of instances per site for distance learning
//const int TEST_NUM = 30; //number of instances per site for kNN training/testing
//int OPENTEST_NUM = 0; //number of open instances for kNN training/testing

// NEIGBOUR_NUM is given as a parameter to the prediction function.
//int NEIGHBOUR_NUM = 2; //number of neighbors for kNN
int RECOPOINTS_NUM = 5; //number of neighbors for distance learning

//Algorithmic Parameters
float POWER = 0.1; //not used in this code

float dist(float* feat1, float* feat2, float* weight, float power) {
	float toret = 0;
	for (int i = 0; i < FEAT_NUM; i++) {
		if (feat1[i] != -1 and feat2[i] != -1) {
			toret += weight[i] * abs(feat1[i] - feat2[i]);
		}
	}
	return toret;
}

/* Recommends the best weights using the training
 * set (feat, labels).
 */
/* The function used to be:
 * //void alg_recommend2(float** feat, float* weight, int start, int end) {
 */
extern "C"
void recommend_weight(float **feat, int *labels, int n, float *weight,
                      int start, int end) {
    float *distlist = new float[n];
    int *recogoodlist = new int[RECOPOINTS_NUM];
    int *recobadlist = new int[RECOPOINTS_NUM];

	for (int i = start; i < end; i++) {
		printf("\rLearning distance... %d (%d-%d)", i, start, end);
		fflush(stdout);
        int cur_site = labels[i];

		float pointbadness = 0;
		float maxgooddist = 0;

		for (int k = 0; k < n; k++) {
			distlist[k] = dist(feat[i], feat[k], weight, POWER);
		}

		float max = *max_element(distlist, distlist+n);

        distlist[i] = max;
        for (int k = 0; k < RECOPOINTS_NUM; k++) {
            /* Find the nearest neighbour to feat[i] having the same label. */
            int ind = min_element(distlist, distlist+n) - distlist;
            while (labels[ind] != cur_site) {
                distlist[ind] = max;
                ind = min_element(distlist, distlist+n) - distlist;
            }
            /* Keep track of the maximum distance of an object with the
             * same label.
             */
            if (distlist[ind] > maxgooddist)
                maxgooddist = distlist[ind];
            /* Remove the neighbour, record it, and repeat. */
            distlist[ind] = max;
            recogoodlist[k] = ind;
        }

        /* Set to max the distance to the neighbours of the same label. */
        for (int k = 0; k < n; k++)
            if (labels[k] == cur_site)
                distlist[k] = max;

        /* Find the nearest neighbour to feat[i] having a different label. */
        for (int k = 0; k < RECOPOINTS_NUM; k++) {
            int ind = min_element(distlist, distlist+n) - distlist;
            /* Sum number of bad points (up to RECOPOINTS_NUM). */
            if (distlist[ind] <= maxgooddist)
                pointbadness += 1;
            /* Remove the neighbour, record it, and repeat. */
            distlist[ind] = max;
            recobadlist[k] = ind;
        }

		pointbadness /= float(RECOPOINTS_NUM);
		pointbadness += 0.2;

		float* featdist = new float[FEAT_NUM];
		for (int f = 0; f < FEAT_NUM; f++) {
			featdist[f] = 0;
		}
		int* badlist = new int[FEAT_NUM];
		int minbadlist = 0;
		int countbadlist = 0;
		//printf("%d ", badlist[3]);
		for (int f = 0; f < FEAT_NUM; f++) {
			/* Bug note. In the original code, the following was:
             *      if (weight[f] == 0) badlist[f] == 0;
             */
			if (weight[f] == 0) badlist[f] = 0;
			else {
			float maxgood = 0;
			int countbad = 0;
			for (int k = 0; k < RECOPOINTS_NUM; k++) {
				float n = abs(feat[i][f] - feat[recogoodlist[k]][f]);
				if (feat[i][f] == -1 or feat[recobadlist[k]][f] == -1) 
					n = 0;
				if (n >= maxgood) maxgood = n;
			}
			for (int k = 0; k < RECOPOINTS_NUM; k++) {
				float n = abs(feat[i][f] - feat[recobadlist[k]][f]);
				if (feat[i][f] == -1 or feat[recobadlist[k]][f] == -1) 
					n = 0;
				featdist[f] += n;
				if (n <= maxgood) countbad += 1;
			}
			badlist[f] = countbad;
			if (countbad < minbadlist) minbadlist = countbad;	
			}
		}

		for (int f = 0; f < FEAT_NUM; f++) {
			if (badlist[f] != minbadlist) countbadlist += 1;
		}
		int* w0id = new int[countbadlist];
		float* change = new float[countbadlist];

		int temp = 0;
		float C1 = 0;
		float C2 = 0;
		for (int f = 0; f < FEAT_NUM; f++) {
			if (badlist[f] != minbadlist) {
				w0id[temp] = f;
				change[temp] = weight[f] * 0.01 * badlist[f]/float(RECOPOINTS_NUM) * pointbadness;
				C1 += change[temp] * featdist[f];
				C2 += change[temp];
				weight[f] -= change[temp];
				temp += 1;
			}
		}

		float totalfd = 0;
		for (int f = 0; f < FEAT_NUM; f++) {
			if (badlist[f] == minbadlist and weight[f] > 0) {
				totalfd += featdist[f];
			}
		}

		for (int f = 0; f < FEAT_NUM; f++) {
			if (badlist[f] == minbadlist and weight[f] > 0) {
				weight[f] += C1/(totalfd);
			}
		}

		delete[] featdist;
		delete[] w0id;
		delete[] change;
		delete[] badlist;
	}


	for (int j = 0; j < FEAT_NUM; j++) {
		if (weight[j] > 0)
			weight[j] *= (0.9 + (rand() % 100) / 500.0);
	}
	printf("\n");
	delete[] distlist;
	delete[] recobadlist;
	delete[] recogoodlist;



}

/* Returns predictions of the k-NN attack using training data
 * (feat_train, labels_train) against test data feat_test.
 * It uses trained weights weight. It returns the results in
 * an array of int, predictions, which should be initialised before.
 * Array predictions will contain one of {0, ..., n_labels-1} for
 * labels, or "-1" for non-monitored pages (i.e., where there is no
 * consensus among the neighbours).
 */
/* Used to be:
 *  void accuracy(float** closedfeat, float* weight, float** openfeat, float & tp, float & tn) {
 */
extern "C"
void knn_predict(float **feat_train, int *labels_train, float **feat_test,
                 int n_train, int n_test, int n_labels, int n_neighbours,
                 float *weight, int *predictions) {

    float *distlist = new float[n_train];
    int *classlist = new int[n_labels + 1]; /* Count of neighbours per label */

    printf("Running k-NN attack for k=%d\n", n_neighbours);


    for (int is = 0; is < n_test; is ++) {

        /* Reset classlist */
        for (int i = 0; i < n_labels+1; i++) {
            classlist[i] = 0;
        }

        int maxclass = 0;       /* Most frequent class among neighbours */
        /* NOTE: the original code computes the distance also from the
         * test instances. But this is incorrect, because the adversary
         * does not have access to the test instances' labels.
         * We should thus only compute distances from training instances.
         */
        for (int at = 0; at < n_train; at++) {
            distlist[at] = dist(feat_test[is], feat_train[at], weight, POWER);
        }

        float max = *max_element(distlist, distlist+n_train);
        /* NOTE: the original code now sets the distance to the current
         * instance to max. There's no need to do that here, because
         * distlist only contains distances from training instances
         * (which the current instance is not).
         */
        /* Find the n_neighbours neighbours */
        for (int i = 0; i < n_neighbours; i++) {
            int ind = find(distlist, distlist+n_train,
                           *min_element(distlist, distlist+n_train)) - distlist;
            /* NOTE: there should be no need to do the following
             * (commented out), because there
             * is no distance from open world instances. In fact, by
             * definition, the adversary should not observe them if not
             * _individually_ (unless we were assuming some sort of on-line
             * learning, which we are not) during the test phase as test
             * objects..
             */
			//int classind = 0;
			//if (ind < SITE_NUM * TEST_NUM) {
			//	classind = ind/TEST_NUM;
			//}
			//else {
			//	classind = SITE_NUM;
			//}
			//classlist[classind] += 1;

            /* Now the class is certainly from the training set */
            int classind = labels_train[ind];
            classlist[classind] += 1;

			if (classlist[classind] > maxclass) {
				maxclass = classlist[classind];
			}
            /* Exclude the neighbour we just found from the distances list */
			distlist[ind] = max;
		}

        /* Predicts the label or "-1" ("non-monitored") if no consensus.
         * There is consensus ONLY IF one of classlist entries is
         * equal to n_neighbours. This also means that if there
         * is consensus then only one entry of classlist is
         * non-zero (precisely, it is equal to n_neighbours).
         */
        int predicted_label = -1;
		for (int i = 0; i < n_labels+1; i++) {
			if (classlist[i] == n_neighbours) {
                predicted_label = i;
                break;
			}
		}

        predictions[is] = predicted_label;

        /* NOTE on the original code, which is commented out as we
         * compute the accuracy on the callee function:
         * if more than one class was predicted, the accuracy would be
         * divided by the number of classes. However this does not
         * follows what the original paper says, i.e.:
         *     "a point should be classified as a monitored page only
         *     if all k neighbours agree on which page it is, and
         *     otherwise it will be classified as a non-monitored page".
         * Nevertheless, this (i.e., countclass > 1) should never happen
         * because: if there is consensus then only one entry of classlist
         * is non-zero; if there is no consensus, the original code
         * sets the entries 0..n_labels of classlist to 0, and
         * classlist[n_labels+1] = 1 and maxclass = 1, meaning that
         * countclass will be 1 (only "non monitored" class n_labels+1 is
         * non-zero).
         */

		//float thisacc = 0;
		//if (hasconsensus == 0) {
		//	for (int i = 0; i < SITE_NUM; i++) {
		//		classlist[i] = 0;
		//	}
		//	classlist[SITE_NUM] = 1;
		//	maxclass = 1;
		//}

		//for (int i = 0; i < SITE_NUM+1; i++) {
		//	if (classlist[i] == maxclass) {
		//		countclass += 1;
		//		if (i == trueclass) {
		//			hascorrect = 1;
		//		}
		//	}
		//}

		//if (hascorrect == 1) {
		//	thisacc = 1.0/countclass;
		//}
		//if (trueclass == SITE_NUM) {
		//	tn += thisacc;
		//}
		//else { 
		//	tp += thisacc;
		//}
		
	}

	delete[] distlist;
	delete[] classlist;
}
