
# PyPi upload

Package a Python Package/ version bump See: https://packaging.python.org/tutorials/packaging-projects/

1. Update `setup.py` to new version number
2. Commit this change
3. Tag and upload

```bash
git clone git@github.com:fgnt/lazy_dataset.git lazy_dataset_pypi_version
cd lazy_dataset_pypi_version
pip install --upgrade setuptools
pip install --upgrade wheel
pip install --upgrade twine
# pip install --upgrade bleach html5lib  # some versions do not work
git tag # Lists existing tags
git tag -a 0.0.1 -m "Pypi version update"
git tag # Lists existing tags
git push origin --tags
python setup.py sdist bdist_wheel
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
twine upload dist/*
```
