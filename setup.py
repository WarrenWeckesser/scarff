# Copyright © 2021 Warren Weckesser

import setuptools
from setuptools import setup
from os import path


def get_scarff_version():
    """
    Find the value assigned to __version__ in __init__.py.

    This function assumes that there is a line of the form

        __version__ = "version-string"

    in scarff/__init__.py.  It returns the string version-string,
    or None if such a line is not found.
    """
    with open(path.join("scarff", "__init__.py"), "r") as f:
        for line in f:
            s = [w.strip() for w in line.split("=", 1)]
            if len(s) == 2 and s[0] == "__version__":
                return s[1][1:-1]


# Get the long description from README.rst.
_here = path.abspath(path.dirname(__file__))
with open(path.join(_here, 'README.rst')) as f:
    _long_description = f.read()

setup(
    name='scarff',
    version=get_scarff_version(),
    author='Warren Weckesser',
    description="Write NumPy arrays and SciPy sparse matrices to ARFF files.",
    long_description=_long_description,
    license="MIT",
    url="https://github.com/WarrenWeckesser/scarff",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
    ],
    keywords="numpy scipy ARFF",
)
