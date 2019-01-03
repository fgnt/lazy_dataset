from .database import (
    DictDatabase,
    JsonDatabase,
    KaldiDatabase,
    HybridASRKaldiDatabaseTemplate
)

__all__ = [
    "database",
    "iterator",
    "helper",
    "keys",

    "ami",
    "audio_set",
    "chime",
    "chime5",
    "dcase_2017",
    "dcase_2018",
    "german_speechdata_package_v2",
    "librispeech",
    "merl_mixtures",
    "noisex92",
    "reverb",
    "tidigits",
    "timit",
    "wsj",
    "wsj_bss",
    "wsj_mc",
    "wsj_voicehome",
]

# Lazy import all subpackages
# Note: define all subpackages in __all__
import sys
import pkgutil
import operator
import importlib.util

_available_submodules = list(map(
    operator.itemgetter(1),
    pkgutil.iter_modules(__path__)
))


class _LazySubModule(sys.modules[__name__].__class__):
    # In py37 is the class is not nessesary and __dir__ and __getattr__ are enough
    # https://snarky.ca/lazy-importing-in-python-3-7/

    def __dir__(self):
        ret = super().__dir__()
        return [*ret, *_available_submodules]

    def __getattr__(self, item):
        if item in _available_submodules:
            import importlib
            return importlib.import_module(f'{__package__}.{item}')
        else:
            return super().__getattr__(item)


sys.modules[__name__].__class__ = _LazySubModule

del sys, pkgutil, operator, importlib, _LazySubModule
