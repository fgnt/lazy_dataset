#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Original: https://github.com/kennethreitz/setup.py

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'lazy_dataset'
DESCRIPTION = 'Process large datasets as if it was an iterable.'
URL = 'https://github.com/fgnt/lazy_dataset'
EMAIL = 'boeddeker@nt.upb.de'
AUTHOR = 'Christoph Boeddeker'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.0.14'

# What packages are required for this module to be executed?
REQUIRED = [
    'numpy',
]

# What packages are optional?
EXTRAS = {
    'cache': ['humanfriendly', 'psutil', 'diskcache'],
    'test': [
        'paderbox',
        'mock',
        'dill',  # special backend for prefetch
        'pathos',  # special backend for prefetch
    ],
    'cli': [
        # When using Python 2.7, please install IPython 5.x LTS Long Term Support version.
        # Python 3.3 and 3.4 were supported up to IPython 6.x.
        # Python 3.5 was supported with IPython 7.0 to 7.9.
        # Python 3.6 was supported with IPython up to 7.16.
        # Python 3.7 was still supported with the 7.x branch.
        "IPython<8.0; python_version=='3.7'",  # IPython 8.0-8.12 supports Python 3.8 and above, following NEP 29.
        "IPython<8.13.0; python_version=='3.8'",  # IPython 8.13+ supports Python 3.9 and above, following NEP 29.
        "IPython; python_version>='3.9'",
        'paderbox',
    ],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')
        
        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require={
        **EXTRAS,
        'all': set().union(*(x for x in EXTRAS.values())),
    },
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
