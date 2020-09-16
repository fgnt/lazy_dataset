import json
import weakref
from pathlib import Path
import typing
import copy

import lazy_dataset


class Database:
    """Base class for databases.

    This class is abstract!
    """

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

        Under the key `datasets` are the datasets, where each dataset
        is a dictionary that mapps from an example_id to the example. Is is
        assumed that the example is a dictionary and does not contain the keys
        `dataset` and `example_id`. These keys are added in the `get_dataset`
        method.

        Under the key `alias` are datasets listed that group multiple datasets
        to a new dataset.

        Beside the keys `datasets` and `alias` may exist further keys, they
        are ignored in the base class, but may be used in inherited database
        classes.

        In case of an speech audio mixture an example may look as follows:

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

        The example does not contain binary data (e.g. audio signals).
        Instead it contains the paths to them. In this may big databases fit
        into the memory. In a later map function the binary data can be loaded.

        We recommend to use absolute paths to the data, so that working with
        the example is as easy as possible.
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

    def get_examples(self, dataset_name):
        """
        Get examples dict for a certain dataset name. example_id and dataset
        name are added to each example.

        Do not make inplace manipulations of the returned dictionary!!!

        Args:
            dataset_name: the name of the requested dataset

        Returns: a dictionary with examples from the requested dataset

        """
        try:
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
            else:
                examples = {**self.data['datasets'][dataset_name]}
        except KeyError:
            import difflib
            similar = difflib.get_close_matches(
                dataset_name, self.dataset_names, n=5, cutoff=0)
            raise KeyError(dataset_name, f'close_matches: {similar}', self)

        if len(examples) == 0:
            # When somebody need empty datasets, add an option to this
            # function to allow empty datasets.
            raise RuntimeError(
                f'The requested dataset {dataset_name!r} is empty. '
            )

        for example_id in examples.keys():
            examples[example_id] = {
                **examples[example_id],
                'example_id': example_id,
                'dataset': dataset_name,
            }
        return examples

    def get_dataset(self, name=None):
        """Return a single lazy dataset over specified datasets.

        Adds the example_id and dataset_name to each example dict.

        This function should never be overwritten.

        Args:
            name: list or str specifying the datasets of interest.
            If None an exception msg is raised that shows all available names.
            When the requested dataset does not exist, the closest matches are
            displayed in the exception msg.

        Returns:
            A lazy dataset.
        """
        return self._get_dataset(name)

    def _get_dataset(self, name=None):
        """
        The private get_dataset function allows for databases to overwrite
        get_dataset and adding a map function without recursively calling
        the map function in case of multiple dataset names.
        """
        if name is None:
            raise TypeError(
                f'Missing dataset_name, use e.g.: {self.dataset_names}'
            )

        if isinstance(name, str):
            pass
        elif isinstance(name, typing.Iterable) and not isinstance(name, dict):
            datasets = [self._get_dataset(n) for n in name]
            return lazy_dataset.concatenate(*datasets)
        else:
            raise TypeError(
                f'Argument type {type(name)} of {name} is not allowed!'
                f'Expected are str, list or tuple.'
            )

        # Resulting dataset is immutable anyway due to pickle in
        # `lazy_dataset.from_dict`. This code here avoids to store the
        # resulting dataset more than once in memory. Discuss with CBJ for
        # details.
        try:
            return self._dataset_weak_ref_dict[name]
        except KeyError:
            pass

        examples = self.get_examples(name)
        ds = lazy_dataset.from_dict(examples, name=name)

        self._dataset_weak_ref_dict[name] = ds

        return ds


class DictDatabase(Database):
    def __init__(
            self,
            database_dict: typing.Union[
                dict, typing.List[dict], typing.Tuple[dict]],
            *database_dicts: dict,
    ):
        """
        A simple database class intended to hold a given database_dict.

        Args:
            database_dict: A pickle serializeable database dictionary.
        """
        if isinstance(database_dict, (list, tuple)):
            assert not database_dicts, (
                f'Passing multiple lists of dicts is not supported. '
                f'Use either DictDatabase([dict1, dict2, ...]) or '
                f'DictDatabase(dict1, dict2, ...).',
                database_dict, database_dicts
            )
        else:
            database_dict = [database_dict] + list(database_dicts)
        assert len(database_dict) > 0, 'At least one database dict is required'
        self._data = _merge_database_dicts(*database_dict)
        super().__init__()

    @property
    def data(self):
        return self._data


class JsonDatabase(Database):
    def __init__(
            self,
            json_path: typing.Union[
                str, Path,
                typing.List[typing.Union[str, Path]],
                typing.Tuple[typing.Union[str, Path]]],
            *json_paths: typing.Union[str, Path],
    ):
        """

        Args:
            json_path: One or multiple paths to database JSONs. Supports both
                `JsonDatabase('json1.json', 'json2.json')` and
                `JsonDatabase(['json1.json', 'json2.json')`.

        """
        if isinstance(json_path, (list, tuple)):
            assert not json_paths, (
                f'Passing multiple lists of json paths is not supported. '
                f'Use either JsonDatabase(["json1.json", "json2.json", ...]) '
                f'or JsonDatabase("json1.json", "json2.json", ...).',
                json_path, json_paths
            )
            self._json_path = list(json_path)
        else:
            self._json_path = [json_path] + list(json_paths)

        assert len(self._json_path) > 0, (
            'At least one database JSON is required!'
        )
        super().__init__()

    _data = None

    @property
    def data(self):
        if self._data is None:
            self._data = _merge_database_dicts(*[
                json.loads(Path(path).expanduser().read_text())
                for path in self._json_path
            ])

        return self._data

    def __repr__(self):
        return f'{type(self).__name__}({self._json_path!r})'


def _merge_database_dicts(*database_dicts):
    # Copy to prevent writes to the original dict
    result = copy.deepcopy(database_dicts[0])

    for database_dict in database_dicts[1:]:
        assert not set(database_dict.keys()) - {'datasets', 'alias'}, (
            f'All but the first database are only allowed to have keys'
            f'"datasets" and "alias".'
        )

        # Get dataset names to check for duplicates
        dataset_names = set(result['datasets'].keys()) | set(
            result.get('alias', {}).keys())

        # Check and update "datasets"
        duplicate_keys = set(database_dict['datasets'].keys()).intersection(
            dataset_names)
        assert not duplicate_keys, (
            f'Found duplicate dataset names in databases! {duplicate_keys}'
        )
        result['datasets'].update(database_dict['datasets'])

        # Check and update "alias"
        if 'alias' in database_dict:
            duplicate_keys = set(database_dict['alias'].keys()).intersection(
                dataset_names)
            assert not duplicate_keys, (
                f'Found duplicate alias names in databases! '
                f'{duplicate_keys}'
            )

            result['alias'].update(database_dict['alias'])

    return result
