"""
ToDo: Fix this text.

The reader is part of the new database concept 2017.

The task of the reader is to take a database JSON and an dataset identifier as
an input and load all meta data for each observation with corresponding
numpy arrays for each time signal (non stacked).

An example ID often stands for utterance ID. In case of speaker mixtures,
it replaces mixture ID. Also, in case of event detection utterance ID is not
very adequate.

The JSON file is specified as follows:

datasets:
    <dataset name 0>
        <unique example id 1> (unique within a dataset)
            audio_path:
                speech_source:
                    <path to speech of speaker 0>
                    <path to speech of speaker 1>
                observation:
                    blue_array: (a list, since there are no missing channels)
                        <path to observation of blue_array and channel 0>
                        <path to observation of blue_array and channel 0>
                        ...
                    red_array: (special case for missing channels)
                        c0: <path to observation of red_array and channel 0>
                        c99: <path to observation of red_array and channel 99>
                        ...
                speech_image:
                    ...
            speaker_id:
                <speaker_id for speaker 0>
                ...
            gender:
                <m/f>
                ...
            ...

Make sure, all keys are natsorted in the JSON file.

Make sure, the names are not redundant and it is clear, which is train, dev and
test set. The names should be as close as possible to the original database
names.

An observation/ example has information according to the keys file.

If a database does not have different arrays, the array dimension can be
omitted. Same holds true for the channel axis or the speaker axis.

Num samples (Used for bucket boundaries in batching)
- If num_samples has the same value for all signals it is supposed to be a
    scalar, this holds for close to all databases.
    (e.g. num_samples=42...)
- If one observation has different number of samples, num_samples is a
    dictorionary with the same structure as audio_path/observation.
    (e.g. num_samples=dict(observation=dict(U1=42...)), Chime5, etc...)
- At the moment we keep the dict structure of num_samples in WSJ_BSS for
    backward compatibility for LD

The different axis have to be natsorted, when they are converted to numpy
arrays. Skipping numbers (i.e. c0, c99) is database specific and is not handled
by a generic implementation.

If audio paths are a list, they will be stacked to a numpy array. If it is a
dictionary, it will become a dictionary of numpy arrays.

If the example IDs are not unique in the original database, the example IDs
are made unique by prefixing them with the dataset name of the original
database, i.e. dt_simu_c0123.
"""
import json
import weakref
from pathlib import Path

import lazy_dataset


class Database:
    """Base class for databases.

    This class is abstract!"""

    def __init__(self):
        self._dataset_weak_ref_dict = weakref.WeakValueDictionary()

    @property
    def data(self):
        """
        Returns a nested dictionary with the following structure:
        {
            'datasets': {
                <dataset_name_1>: {
                    <example_id>: {...},
                    ...
                },
                ...
            },
            'alias': {
                <dataset_name>: [<dataset_name_1>, <dataset_name_2>],
                ...
            }
        }

        The under the key `datasets` are the datasets, where each dataset
        is a dictionary from example_id to example. Is is assumed that the
        example is a dictionary and does not contain the keys `dataset` and
        `example_id`. These keys are added in the `get_dataset` method.

        Under the key `alias` are datasets listed that group multiple datasets
        to a new dataset.

        Beside the keys `datasets` and `alias` may exist further keys, they
        are ignored in the base class, but may be used in inherited database
        classes.
        """
        raise NotImplementedError(
            f'Override this property in {self.__class__.__name__}!')

    @property
    def dataset_names(self):
        """
        A tuple of all available dataset names, i.e. the keys of
        `data['datasets']` and `data['alias']`.
        """
        return tuple(
            self.data['datasets'].keys()
        ) + tuple(
            self.data.get('alias', {}).keys()
        )

    def _get_dataset_examples_from_data(self, dataset_name):
        if dataset_name in self.data.get('alias', []):
            dataset_names = self.data['alias'][dataset_name]
            examples = {}
            for name in dataset_names:
                examples_new = self.data['datasets'][name]
                intersection = set.intersection(
                    set(examples.keys()),
                    set(examples_new.keys()),
                )
                assert len(intersection) == 0, intersection
                examples = {**examples, **examples_new}
            return examples
        else:
            return self.data['datasets'][dataset_name]

    def get_dataset(self, name=None):
        """Return a single lazy dataset over specified datasets.

        Adds the example_id and dataset_name to each example dict.

        This function should never be overwritten.

        :param names: list or str specifying the datasets of interest.
            If None an iterator over the complete databases will be returned.
        :return:
        """
        if name is None:
            raise TypeError(
                f'Missing dataset_name, use e.g.: {self.dataset_names}'
            )

        if isinstance(name, (tuple, list)):
            datasets = [self.get_dataset(n) for n in name]
            return lazy_dataset.concatenate(*datasets)

        # Resulting dataset is immutable anyway due to pickle a few lines
        # further down. This code here avoids to store the resulting
        # dataset more than once in memory. Discuss with CBJ for details.
        try:
            return self._dataset_weak_ref_dict[name]
        except KeyError:
            pass

        try:
            examples = self._get_dataset_examples_from_data(name)
        except KeyError:
            import difflib
            similar = difflib.get_close_matches(
                name, self.dataset_names, n=5, cutoff=0)
            raise KeyError(name, f'close_matches: {similar}', self)

        if len(examples) == 0:
            # When somebody need empty datasets, add an option to this
            # function to allow empty datasets.
            raise RuntimeError(
                f'The requested dataset {name!r} is empty. '
            )

        for example_id in examples.keys():
            examples[example_id]['example_id'] = example_id
            examples[example_id]['dataset'] = name

        ds = lazy_dataset.from_dict(examples)

        self._dataset_weak_ref_dict[name] = ds

        return ds


class DictDatabase(Database):
    def __init__(self, database_dict: dict):
        """
        A simple database class intended to hold a given database_dict.

        Args:
            database_dict: A pickle serializeable database dictionary.
        """
        self._data = database_dict
        super().__init__()

    @property
    def data(self):
        return self._data


class JsonDatabase(Database):
    def __init__(self, json_path: [str, Path]):
        """

        Args:
            json_path: path to database JSON

        """
        self._json_path = json_path
        super().__init__()

    _data = None

    @property
    def data(self):
        if self._data is None:
            path = Path(self._json_path).expanduser()

            with path.open() as fd:
                self._data = json.load(fd)
        return self._data

    def __repr__(self):
        return f'{type(self).__name__}({self._json_path!r})'
