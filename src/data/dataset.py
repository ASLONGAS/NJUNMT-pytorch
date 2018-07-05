import collections
import random
import tempfile
import os


from src.utils.logging import INFO
from .vocabulary import _Vocabulary

__all__ = [
    'TextDataset',
    'ZipDataset'
]


def _shuffle(*path):
    f_handles = [open(p) for p in path]

    # Read all the data
    lines = []
    for l in f_handles[0]:
        line = [l.strip()] + [ff.readline().strip() for ff in f_handles[1:]]
        lines.append(line)

    # close file handles
    [f.close() for f in f_handles]

    # random shuffle the data
    INFO('Shuffling data...')
    random.shuffle(lines)
    INFO('Done.')

    # Set up temp files
    f_handles = []
    for p in path:
        _, filename = os.path.split(p)
        f_handles.append(tempfile.TemporaryFile(prefix=filename + '.shuf', dir="/tmp/", mode="a+"))

    for line in lines:
        for ii, f in enumerate(f_handles):
            print(line[ii], file=f)

    # release memory
    lines = []

    # Reset file handles
    [f.seek(0) for f in f_handles]

    return tuple(f_handles)


class Dataset(object):
    """
    In ```Dataset``` object, you can define how to read samples from different formats of
    raw data, and how to organize these samples. There are some things you need to override:
        - In ```n_fields``` you should define how many fields in one sample.
        - In ```__len__``` you should define the capacity of your dataset.
        - In ```_data_iter``` you should define how to read your data, using shuffle or not.
        - In ```_apply``` you should define how to transform your raw data into some kind of format that can be
        computation-friendly.
    """

    def __init__(self, *args, **kwargs):
        pass

    @property
    def n_fields(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _apply(self, *lines):
        """ Do some processing on the raw input of the dataset.

        Return ```None``` when you don't want to output this line.

        Args:
            lines: A tuple representing one line of the dataset, where ```len(lines) == self.n_fields```

        Returns:
            A tuple representing the processed output of one line, whose length equals ```self.n_fields```
        """
        raise NotImplementedError

    def _data_iter(self, shuffle):
        """ Generate file handles of datasets.

        Always return a tuple of handles.
        """
        raise NotImplementedError

    def _not_empty(self, *lines):

        if len([1 for l in lines if l is None]) == 0:
            return True
        else:
            return False

    def data_iter(self, shuffle=False):

        f_handles = self._data_iter(shuffle=shuffle)

        if not isinstance(f_handles, collections.Sequence):
            f_handles = [f_handles]

        for lines in zip(*f_handles):

            lines = self._apply(*lines)

            if self._not_empty(*lines):
                yield lines

        [f.close() for f in f_handles]


class TextDataset(Dataset):
    """
    ```TextDataset``` is one kind of dataset each line of which is one sample. There is only one field each line.
    """

    def __init__(self,
                 data_path,
                 vocabulary: _Vocabulary,
                 max_len=-1,
                 shuffle=False
                 ):

        super(TextDataset, self).__init__()

        self._data_path = data_path
        self._vocab = vocabulary
        self._max_len = max_len
        self.shuffle = shuffle

        with open(self._data_path) as f:
            self.num_lines = sum(1 for _ in f)

    @property
    def n_fields(self):
        return 1

    def __len__(self):
        return self.num_lines

    def _data_iter(self, shuffle):

        if shuffle:
            return _shuffle(self._data_path)
        else:
            return open(self._data_path)

    def _apply(self, *lines):
        """
        Process one line

        :type line: str
        """

        line = self._vocab.tokenize(lines[0])

        line = [self._vocab.token2id(w) for w in line]

        if 0 < self._max_len < len(line):
            return None

        return line


class ZipDataset(Dataset):
    """
    ```ZipDataset``` is a kind of dataset which is the combination of several datasets. The same line of all
    the datasets consist on sample. This is very useful to build dataset such as parallel corpus in machine
    translation.
    """

    def __init__(self, *datasets, shuffle=False):
        """
        """
        super(ZipDataset, self).__init__()
        self.shuffle = shuffle
        self.datasets = datasets

    @property
    def n_fields(self):
        return len(self.datasets)

    def __len__(self):
        return len(self.datasets[0])

    def _data_iter(self, shuffle):

        if shuffle:
            return _shuffle(*[ds._data_path for ds in self.datasets])
        else:
            return [open(ds._data_path) for ds in self.datasets]

    def _apply(self, *lines):
        """
        :type dataset: TextDataset
        """

        outs = [d._apply(l) for d, l in zip(self.datasets, lines)]

        return outs  # (line_1, line_2, ..., line_n)
