import sys
import time
import contextlib
import pickle as pkl
import json
import numpy as np

from . import nest

__all__ = [
    'batch_open',
    'GlobalNames',
    'Timer',
    'Collections',
    'sequence_mask',
    'build_vocab_shortlist',
    'to_gpu'
]


# ================================================================================== #
# File I/O Utils

@contextlib.contextmanager
def batch_open(refs, mode='r'):
    handlers = []
    if not isinstance(refs, (list, tuple)):
        refs = [refs]
    for f in refs:
        handlers.append(open(f, mode))

    yield handlers

    for h in handlers:
        h.close()


class GlobalNames:
    # learning rate variable name
    MY_LEARNING_RATE_NAME = "learning_rate"

    MY_CHECKPOINIS_PREFIX = ".ckpt"

    MY_BEST_MODEL_SUFFIX = ".best.tpz"

    MY_BEST_OPTIMIZER_PARAMS_SUFFIX = ".best_optim.tpz"

    MY_COLLECTIONS_SUFFIX = ".collections.pkl"

    MY_MODEL_ARCHIVES_SUFFIX = ".archives.pkl"

    USE_GPU = False

    SEED = 314159


time_format = '%Y-%m-%d %H:%M:%S'


class Timer(object):
    def __init__(self):
        self.t0 = 0

    def tic(self):
        self.t0 = time.time()

    def toc(self, format='m:s', return_seconds=False):
        t1 = time.time()

        if return_seconds is True:
            return t1 - self.t0

        if format == 's':
            return '{0:d}'.format(t1 - self.t0)
        m, s = divmod(t1 - self.t0, 60)
        if format == 'm:s':
            return '%d:%02d' % (m, s)
        h, m = divmod(m, 60)
        return '%d:%02d:%02d' % (h, m, s)


class Collections(object):
    """Collections for logs during training.

    Usually we add loss and valid metrics to some collections after some steps.
    """
    _MY_COLLECTIONS_NAME = "my_collections"

    def __init__(self, kv_stores=None, name=None):

        self._kv_stores = kv_stores if kv_stores is not None else {}

        if name is None:
            name = Collections._MY_COLLECTIONS_NAME
        self._name = name

    def load(self, archives):

        if self._name in archives:
            self._kv_stores = archives[self._name]
        else:
            self._kv_stores = []

    def add_to_collection(self, key, value):
        """
        Add value to collection

        :type key: str
        :param key: Key of the collection

        :param value: The value which is appended to the collection
        """
        if key not in self._kv_stores:
            self._kv_stores[key] = [value]
        else:
            self._kv_stores[key].append(value)

    def export(self):
        return {self._name: self._kv_stores}

    def get_collection(self, key):
        """
        Get the collection given a key

        :type key: str
        :param key: Key of the collection
        """
        if key not in self._kv_stores:
            return []
        else:
            return self._kv_stores[key]

    @staticmethod
    def pickle(path, **kwargs):
        """
        :type path: str
        """
        archives_ = dict([(k, v) for k, v in kwargs.items()])

        if not path.endswith(".pkl"):
            path = path + ".pkl"

        with open(path, 'wb') as f:
            pkl.dump(archives_, f)

    @staticmethod
    def unpickle(path):
        """:type path: str"""

        with open(path, 'rb') as f:
            archives_ = pkl.load(f)

        return archives_


def sequence_mask(seqs_length):
    maxlen = np.max(seqs_length)

    row_vector = np.arange(maxlen)

    mask = row_vector[None, :] < np.expand_dims(seqs_length, -1)

    return mask.astype('float32')


def build_vocab_shortlist(shortlist):
    shortlist_ = nest.flatten(shortlist)

    shortlist_ = sorted(list(set(shortlist_)))

    shortlist_np = np.array(shortlist_).astype('int64')

    map_to_shortlist = dict([(wid, sid) for sid, wid in enumerate(shortlist_np)])
    map_from_shortlist = dict([(item[1], item[0]) for item in map_to_shortlist.items()])

    return shortlist_np, map_to_shortlist, map_from_shortlist


def to_gpu(*inputs):
    return list(map(lambda x: x.cuda(), inputs))
