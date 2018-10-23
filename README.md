*(Repo under construction, as of 23rd October 2018)*


# Website Fingerprinting Evaluation Suite.

This is a suite for evaluating Website Fingerprinting (WF)
attacks and defenses.

It provides:
- a standard interface to use the code from previous attacks/defences,
- a method to estimate security bounds and $(\varepsilon, \Phi)$-privacy of
  a WF defense.

Code from other researchers (acknowledged below) was adapted to fit the API.
With this regard, I tried making as few changes as possible, so as to keep the
results close to the original ones; the changes I made are documented by diff
files.

An introduction for computing security bounds is at
[https://giocher.com/pages/bayes.html](https://giocher.com/pages/bayes.html).

## Installation

```
mkvirtualenv bayes
pip install -r requirements.txt
```

Install weka >=3.8 in ~/$WEKA_VERSION/
Then:
```
java -classpath ~/$WEKA_VERSION/weka.jar weka.core.WekaPackageManager -install-package LibSVM
```
Build attack code by Wang et al.
```
cd code/attacks/wang && make && cd -
```

## Reproducing Experiments

This section should allow you to reproduce the experiments, and should help you
replicating them with your dataset.
I will consider here the dataset by Wang et al. 2014 ("WCN+").

### Getting the WCN+ dataset (and data format explanation)
Download the WCN+ dataset:
```bash
mkdir -p data/WCN+
cd data/WCN+
wget https://www.cse.ust.hk/~taow/wf/data/knndata.zip
unzip knndata.zip
mv batch original
```

The dataset has the following format.
It is a folder containing files $w-$l, with $w being the web page id,
$l indicating the page load.

Each of these files contains, per row:
    
    t_i<tab>s_i

with t_i and s_i indicating respectively time and size of the i-th packet.
The sign of s_i indicates the packet's direction (positive means outgoing).

### Defending the dataset

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

NOTE: most of these scripts assume traces' files are in the format $w-$l,
with w=0..99, l = 0..89 as in the WCN+ dataset.
For decoy-pages, the dataset will need to contain "open world" traces
$w, i=0..8999.
I didn't have the time to change this in Wang's code.

### Extracting features
In order to perform an attack or to compute security bounds you need to
first extract feature vectors ("objects") from traces.
Each page load $w-$l corresponds to a feature vector, and each feature
vector is contained in a file $w-$l.features.

In general, you can extract features for attack $attack as follows:
```bash
python code/extract_features.py --traces $DATASET --out $FEAT_DIR --attack $attack
```
For a list of attacks do:
```bash
python code/extract_features.py -h
```

To extract _all_ feature sets for defended traces in data/$defence for
_all_ defences, do:
```bash
cd code/scripts
bash all_features.sh
```

#### NOTES

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


### Classification (attack)
To evaluate an attack, launch:
```bash
python code/classify.py --features $FEAT_DIR --train 0.8 --test 0.2 --attack $ATTACK --out $OUT_FNAME
```
The output is a json file.

### Computing bounds

Computing bounds is done in two phases, which can be run concurrently.

#### Computing distances
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


#### Computing bounds
Then, you can compute the bounds using:
```bash
python code/bounds.py --distances $DISTANCES --train 0.8 --test 0.2 --out $OUT
```

The output is a json file, which can be read pretty quickly with Python or
with a text editor.


## Hacking
How to add new attacks/defences.

How to add new distance metrics.

## Credits

* attacks/dyer: Kevin P. Dyer (https://github.com/kpdyer/website-fingerprinting)
* attacks/hayes: Jamie Hayes (https://github.com/jhayes14/k-FP)
* attacks/pulls: Tobias Pulls (https://github.com/pylls/go-knn)
* defences/{BuFLO, HTTPOS, tamaraw, traffic_morphing}: Tao Wang (https://cs.uwaterloo.ca/~t55wang/wf.html)
* CUMUL attack's code was inspired by the original by Andriy Panchenko (http://lorre.uni.lu/~andriy/zwiebelfreunde/)
