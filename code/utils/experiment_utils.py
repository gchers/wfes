from sklearn.model_selection import train_test_split

def log(s):
    print '[*] {}'.format(s)

def train_test_closed_world(X, Y, n_train, seed, uniform_labels=True):
    if uniform_labels:
        print "Splitting into training/test with uniform labels"
        stratify = Y
    else:
        print "Splitting into training/test (no uniform labels)"
        stratify = None

    return train_test_split(X, Y, train_size=n_train, random_state=seed,
                            stratify=stratify)
