import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


# IN PROGRESS
class Vocabulary:

    def __init__(self, w2idx, w2vec, idx_misc=None):
        if not idx_misc:
            idx_misc = {'<UNK>': 0,
                        '<PAD>': 1}
        self.idx_misc = idx_misc

        # Attributes for data storage.
        self.w2idx = {**self.idx_misc, **w2idx}
        self.i2w = [word for word, idx in sorted(w2idx.items(),
                                                 key=lambda x: x[1])]
        self.w2vec = w2vec

        # Miscellaneous other attributes.
        self.dim = len(w2vec['the'])
        self.corpus_counts = dict()
        self.embedding_matrix = None

    @classmethod
    def from_glove_file(cls, path, max_lines=float('inf')):
        """Create a new Vocabulary object by loading GloVe vectors from a text
        file.

        Parameters
        -----------
        path: str
            Path to file containing glove vectors.
        max_lines: int, float (optional)
            Loading the GloVe vectors can be slow, so for testing purposes
            it can be helpful to read in a subset. If no value is provided,
            all 400,000 lines in the file will be read in.
        """
        w2idx = dict()
        w2vec = dict()

        with open(path, 'r') as f:
            for i, line in enumerate(f, 2):
                if i > max_lines:
                    break
                word, *values = line.strip().split(' ')
                w2idx[word] = i
                w2vec[word] = np.array(values, dtype=np.float)

        return cls(w2idx, w2vec)

    def save(self, path, verbose=True):
        """Pickle Vocabulary object for later use. We can then quickly load
        the object using torch.load(path), which can be much faster than
        re-computing everything when the vocab size becomes large.

        Parameters
        -----------
        path: str
            Where to save the output file.
        verbose: bool
            If True, print message showing where the object was saved to.
        """
        if verbose:
            print(f'Saving vocabulary to {path}.')
        torch.save(self, path)

    def filter_words(self, tokens, max_words=None, min_freq=0, inplace=False,
                     recompute=False):
        """
        Parameters
        -----------
        tokens: list[str]
            A tokenized list of words in the corpus (must be all lowercase
            when using GloVe vectors). There is no need to hold out test data
            here since we are not using labels.
        max_words: int (optional)
            Provides an upper threshold for the number of words in the
            vocabulary. If no value is passed in, no maximum limit will be
            enforced.
        min_freq: int (optional)
            Provides a lower threshold for the number of times a word must
            appear in the corpus to remain in the vocabulary. If no value is
            passed in, no minimum limit will be enforced.

            Note that we can specify values for both max_words and min_freq
            if desired. If no values are passed in for either, no pruning of
            the vocabulary will be performed.
        inplace: bool
            If True, will change the object's attributes
            (w2idx, w2vec, and i2w) to reflect the newly filtered vocabulary.
            If False, will not change the object, but will simply compute word
            counts and return what the new w2idx would be. This can be helpful
            for experimentation, as we may want to try out multiple values of
            min_freq to decide how many words to keep. After the first call,
            the attribute corpus_counts can also be examined to help determine
            the desired vocab size.
        recompute: bool
            If True, will calculate word counts from the given tokens. If
            False (the default), this will use existing counts if there are
            any.

            The idea is that if we call this method, then realize we want
            to change the corpus, we should calculate new word counts.
            However, if we are simply calling this method multiple times on
            the same corpus while deciding on the exact vocab size we want,
            we should not recompute the word counts.

        Returns
        --------
        dict or None: When called inplace, nothing is returned. When not
        inplace,
        """
        if recompute or not self.corpus_counts:
            self.corpus_counts = Counter(corpus)
        filtered = {word: i for i, (word, freq)
                    in enumerate(self.corpus_counts.most_common(max_words), 2)
                    if freq >= min_freq}
        filtered = {**self.idx_misc, **filtered}

        if inplace:
            # Relies on python3.7 dicts retaining insertion order.
            self.i2w = list(filtered.keys())
            self.w2idx = filtered
            self.w2vec = {word: self.vector(word) for word in filtered}
        else:
            return filtered

    def build_embedding_matrix(self, inplace=False):
        emb = np.zeros((len(self), self.dim))
        for i, word in enumerate(self):
            emb[i] = self.vector(word)

        if inplace:
            self.embedding_matrix = emb
        else:
            return emb

    def idx(self, word):
        """This will map a word (str) to its index (int) in the vocabulary.
        If a string is passed in and the word is not present, a
        value of 1 is returned (the index corresponding to the <UNK> token).

        Parameters
        -----------
        word: str
            A word that needs to be mapped to an integer index.

        Returns
        --------
        int: The index of the given word in the vocabulary.

        Examples
        ---------
        >>> vocab.idx('the')
        2
        """
        return self.w2idx.get(word, 1)

    def vector(self, word):
        """This maps a word to its corresponding embedding vector. If not
        contained in the vocab, a vector of zeros will be returned.

        Parameters
        -----------
        word: str
            A word that needs to be mapped to a vector.

        Returns
        --------
        np.array
        """
        return self.w2vec.get(word, np.zeros(self.dim))

    def __getitem__(self, i):
        """This will map an index (int) to a word (str).

        Parameters
        -----------
        i: int
            Integer index for a word.

        Returns
        --------
        str: Word corresponding to the given index.

        Examples
        ---------
        >>> vocab = Vocabulary(w2idx, w2vec)
        >>> vocab[1]
        '<UNK>'
        """
        return self.i2w[i]

    def __len__(self):
        """Number of words in vocabulary."""
        return len(self.w2idx)

    def __iter__(self):
        for word in self.w2idx.keys():
            yield word

    def __contains__(self, word):
        return word in self.w2idx.keys()


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
