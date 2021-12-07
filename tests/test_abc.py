import pytest
import lazy_dataset
import inspect


subclasses = lazy_dataset.Dataset.__subclasses__()


@pytest.mark.parametrize(
    'method,dataset_cls', [
        (method, cls)
        for method in [
            '__iter__',
            'copy',
            '__len__',
            '__getitem__',
            'keys',
        ]
        for cls in subclasses
    ]
)
def test_signature(method, dataset_cls):
    dataset_sig = inspect.signature(getattr(dataset_cls, method))
    ref_sig = inspect.signature(getattr(lazy_dataset.Dataset, method))

    def remove_annotation(sig: inspect.Signature):
        p: inspect.Parameter
        return sig.replace(
            parameters=[p.replace(annotation=inspect.Parameter.empty)
                        for p in sig.parameters.values()],
            return_annotation=inspect.Signature.empty,
        )

    dataset_sig = remove_annotation(dataset_sig)
    ref_sig = remove_annotation(ref_sig)

    assert dataset_sig == ref_sig
