from collections import Counter
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import torch


# IN PROGRESS: see notebook
class Vocabulary:

    def __init__(self, w2idx, w2vec=None, idx_misc=None, corpus_counts=None):
        """Defines a vocabulary object for NLP problems, allowing users to
        encode text with indices or embeddings.

        Parameters
        -----------
        w2idx: dict[str, int]
            Dictionary mapping words to their integer index in a vocabulary.
            The indices must allow for idx_misc to be added to the dictionary,
            so in the default case this should have a minimum index of 2. If
            a longer idx_misc is passed in, the minimum index would be larger.
        w2vec: dict[str, np.array]
            Dictionary mapping words to their embedding vectors stored as
            numpy arrays. This is optional in case we want to train embeddings
            from scratch.
        idx_misc: dict
            A dictionary mapping non-word tokens to indices. If none is passed
            in, a default version will be used with keys for unknown tokens
            and padding. A customized version might pass in additional tokens
            for repeated characters or all caps, for example.
        """
        if not w2vec:
            w2vec = dict()
            self.dim = 1
        if not idx_misc:
            idx_misc = {'<PAD>': 0,
                        '<UNK>': 1}
        self.idx_misc = idx_misc
        # Check that space has been left for misc keys.
        assert len(idx_misc) == min(w2idx.values())

        # Attributes that store data.
        self.w2idx = {**self.idx_misc, **w2idx}
        self.i2w = [word for word, idx in sorted(self.w2idx.items(),
                                                 key=lambda x: x[1])]
        self.w2vec = w2vec

        # Miscellaneous other attributes.
        self.corpus_counts = dict() if not corpus_counts else corpus_counts
        self.embedding_matrix = None
        if w2vec:
            self.dim = len(w2vec[self[-1]])
        self.w2vec['<UNK>'] = np.zeros(self.dim)

    @classmethod
    def from_glove_file(cls, path, max_lines=float('inf'), idx_misc=None):
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
        misc_len = 2 if not idx_misc else len(idx_misc)

        with open(path, 'r') as f:
            for i, line in enumerate(f, misc_len):
                if i > max_lines:
                    break
                word, *values = line.strip().split(' ')
                w2idx[word] = i
                w2vec[word] = np.array(values, dtype=np.float)

        return cls(w2idx, w2vec, idx_misc)

    @classmethod
    def from_tokens(cls, tokens, idx_misc=None):
        misc_len = 2 if not idx_misc else len(idx_misc)
        counts = Counter(tokens)
        w2idx = {word: i for i, (word, freq)
                 in enumerate(counts.most_common(), misc_len)}
        return cls(w2idx, idx_misc=idx_misc, corpus_counts=counts)

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
        misc_len = len(self.idx_misc)
        if recompute or not self.corpus_counts:
            self.corpus_counts = Counter(tokens)
        filtered = {word: i for i, (word, freq)
                    in enumerate(self.corpus_counts.most_common(max_words),
                                 misc_len)
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
        """Create a 2D numpy array of embedding vectors where row[i]
        corresponds to word i in the vocabulary. This can be used to
        initialize weights in the model's embedding layer.

        Parameters
        -----------
        inplace: bool
            If True, will store the output in the object's embedding_matrix
            attribute. If False (default behavior), will simply return the
            matrix without storing it as part of the object. In the
            recommended case where inplace==False, we can store the output
            in another variable which we can use to initialize the weights in
            Torch, then delete the object and free up memory using
            gc.collect().
        """
        emb = np.zeros((len(self), self.dim))
        for i, word in enumerate(self):
            emb[i] = self.vector(word)

        if inplace:
            self.embedding_matrix = emb
        else:
            return emb

    def idx(self, word):
        """This will map a word (str) to its index (int) in the vocabulary.
        If a string is passed in and the word is not present, the index
        corresponding to the <UNK> token is returned.

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
        return self.w2idx.get(word, self.w2idx['<UNK>'])

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
        return self.w2vec.get(word, self.w2vec['<UNK>'])

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

    def __repr__(self):
        msg = f'Vocabulary({len(self)} words'
        if self.dim > 1:
            msg += f', {self.dim}-D embeddings'
        return msg + ')'


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
