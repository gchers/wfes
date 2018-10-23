**(Repo under construction, as of 23rd October 2018)**


# Website Fingerprinting Evaluation Suite.

This is a suite for evaluating Website Fingerprinting (WF)
attacks and defenses.

It provides:
- a standard interface to use the code from previous attacks/defences,
- a method to estimate security bounds and $(\varepsilon, \Phi)$-privacy of
  a WF defense.

Particularly, it gives a standard interface to use the code from other WF
researchers, who are acknowledged below. Code from other researchers was
adapted to fit the API. With this regard, I tried making as few changes as
possible, so as to keep the results close to the original ones; the changes
I made are documented by diff files.

An introduction to computing security bounds is at
[https://giocher.com/pages/bayes.html](https://giocher.com/pages/bayes.html).

## Installation

This was tested on Alpine Linux 3.8, although I expect it to work for most
BSD and Linux distributions.
Please, open an *issue* should you encounter any problems.

The code works for Python 2.7.

```
mkvirtualenv wfes
pip install -r requirements.txt
```

Install weka >=3.8 in some directory `$WEKA`.
Then install the package `LibSVM`:
```
java -classpath $WEKA/weka.jar weka.core.WekaPackageManager -install-package LibSVM
```

Download, patch, and build attack code.
```
cd code/attacks && make && cd -
```

The following sections should allow you to reproduce the experiments, and
should help you replicating them with your dataset.

## The WCN+ dataset (and data format explanation)

We consider the dataset collected by Wang et al. 2014 ("WCN+").

Download the dataset:
```bash
mkdir -p data/WCN+
cd data/WCN+
wget https://www.cse.ust.hk/~taow/wf/data/knndata.zip
unzip knndata.zip
mv batch original
```

The dataset is constituted of packet sequences corresponding to page loads.
Each packet sequence is contained in a file with name "$W-$L", where $W is the
webpage's id, and $L indicates the page load. For instance, "0-4" is the fourth
page load of webpage 0.

Each of these files contains, per row:
    
    t_i<tab>s_i

with t_i and s_i indicating respectively time and size of the i-th packet.
The sign of s_i indicates the packet's direction (positive means outgoing).
Note: because this dataset represents Tor traffic, where packets' sizes are
fixed, s_i will effectively only indicate the direction, taking value in
{-1, +1}.


## Defending the dataset

You can measure security bounds for any defence.
In this example, we apply the defence directly to the packet sequence files
to morph them; specifically, the defence scripts that follow take as input
a packet sequence file and output a new (morphed) packed sequence file.
If you wish to evaluate other defences, you could simply collect live
network data for them, and estimate security on the generated packet sequence
files -- which should have the format specified above.

Scripts to defend traces can be called as:
```bash
python defend.py --traces $DATASET --out $DEFENDED_DATASET
```

For example:
```bash
python code/defences/WTF-PAD/wtf_pad.py --traces data/WCN+ --out data/WCN+-wtf-pad
```

To defend all:
```bash
DATASET=data/WCN+/original
DST=data/WCN+/

for f in code/defences/*
do
    defence=$(basename $f)
    echo $f/defend.py --traces $DATASET --out $DST/$defence
done | parallel
```

NOTE: most of these scripts assume traces' files are in the format $W-$L,
with W in {0..99}, L in {0..89} as in the WCN+ dataset.
For decoy-pages, the dataset will need to contain "open world" traces
$w, i=0..8999.
It's fairly simple to make this more general, but I didn't have the time to
change this in Wang's code.


## Extracting features

In order to perform an attack or to measure security bounds you need to
first extract feature vectors ("objects") from traces.

Each page load $W-$L corresponds to a feature vector, and, for the purpose
of this document, each feature vector is contained in a file "$W-$L.features".
We create a directory, `$FEAT_DIR`, that will contain the feature vectors.

`cd` into `code/`.

In general, you can extract features for attack $attack as follows:
```bash
python extract_features.py --traces $DATASET --out $FEAT_DIR --attack $attack
```

For instance:
```
python extract_features.py --traces ../data/WCN+/ --attack knn --out ../data/features/knn/
```
where `../data/WCN+/` contains the (possibly defended) packet sequence files,
and `../data/features/knn/' is the output folder that will contain the
resulting feature files.

```
For a list of attacks run:
```bash
python extract_features.py -h
```

The `--type` option can be used to trim the features for specific attacks,
namely k-NN and k-FP; it takes parameter either "knn" or "kfp".
```

To extract _all_ feature sets for defended traces in data/$defence for
_all_ defences, do:
```bash
cd code/scripts
bash all_features.sh
```

### NOTES

*k-NN* features. If the argument "--type knn" is added, weights are applied
to features. This needs to be done for evaluating the attack.
As for computing bounds, this option clearly gives a small advantage
(i.e., bounds are smaller); in the paper, however, I computed bounds without
this option in order to show that the method is robust w.r.t. small
modifications of the feature set.

*k-FP* features. If the argument "--type kfp" is added, features are
extracted using Random Forest as in the paper by Hayes and Danezis.
To my understanding, this is an advantage only in the Open World scenario.
In experments, I did not use this option for attacks nor bounds, as I
observed it produced worse results.


## Classification (attack)
To evaluate an attack, launch:
```bash
python code/classify.py --features $FEAT_DIR --train 0.8 --test 0.2 --attack $ATTACK --out $OUT_FNAME
```
The output is a json file.

## Measuring security

Computing bounds is done in two phases, which can be run concurrently.

### Computing distances
First, you need to compute the pairwise distances between feature vectors:
```bash
python code/compute_distances.py --features $FEAT_DIR --out $OUT
```

FYI, the $OUT file can be opened using dill, should you want to
inspect it.

An alternative to computing distances (and bounds) on feature vectors is to
directly compute them directly on packet sequences (see experiment in
Section 7.4):
```bash
python code/compute_distances --features $TRACES_DIR --sequences --out $OUT
```
Note that this did not produce good results (Fig.5) that is, it seems that
bounds should be computed on feature vectors rather than directly on
packet sequences.


### Computing bounds
Then, you can compute the bounds using:
```bash
python code/bounds.py --distances $DISTANCES --train 0.8 --test 0.2 --out $OUT
```

The output is a json file, which can be read pretty quickly with Python or
with a text editor.


# Hacking

**TODO**
How to add new attacks/defences.

How to add new distance metrics.

# Credits

* attacks/dyer: Kevin P. Dyer (https://github.com/kpdyer/website-fingerprinting)
* attacks/hayes: Jamie Hayes (https://github.com/jhayes14/k-FP)
* attacks/pulls: Tobias Pulls (https://github.com/pylls/go-knn)
* defences/{BuFLO, HTTPOS, tamaraw, traffic_morphing}: Tao Wang (https://cs.uwaterloo.ca/~t55wang/wf.html)
* CUMUL attack's code was inspired by the original by Andriy Panchenko (http://lorre.uni.lu/~andriy/zwiebelfreunde/)
