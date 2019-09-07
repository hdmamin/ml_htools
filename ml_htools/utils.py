from collections import Counter
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


# IN PROGRESS: Vocabulary class (see notebook)


def load_glove(dim, glove_dir):
    """Load glove vectors into a dictionary mapping word to vector.

    Parameters
    -----------
    dim: int
        Size of embedding. One of (50, 100, 200, 300).
    glove_dir: str
        Path to directory containing glove files.

    Returns
    --------
    Dictionary where keys are words and values are {dim}-dimensional ndarrays.
    """
    w2vec = dict()
    path = os.path.join(glove_dir, f'glove.6B.{dim}d.txt')
    with open(path, 'r') as f:
        for row in f:
            items = row.split()
            w2vec[items[0]] = np.array(items[1:], dtype=float)
    return w2vec


def train_val_test_split(x, y, train_p, val_p, state=1, shuffle=True):
    """Wrapper to split data into train, validation, and test sets.

    Parameters
    -----------
    x: pd.DataFrame, np.ndarray
        Features
    y: pd.DataFrame, np.ndarray
        Labels
    train_p: float
        Percent of data to assign to train set.
    val_p: float
        Percent of data to assign to validation set.
    state: int or None
        Int will make the split repeatable. None will give a different random
        split each time.
    shuffle: bool
        If True, randomly shuffle the data before splitting.
    """
    test_p = 1 - val_p / (1 - train_p)
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        train_size=train_p,
                                                        shuffle=shuffle,
                                                        random_state=state)
    x_val, x_test, y_val, y_test = train_test_split(x_test,
                                                    y_test,
                                                    test_size=test_p,
                                                    random_state=state)
    return x_train, x_val, x_test, y_train, y_val, y_test
