# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import re

from pathlib import Path
from setuptools import find_packages, setup

_cwd = Path(os.path.dirname(os.path.realpath(__file__)))

def _get_version(rel_path):
    pattern = re.compile('__version__\s+=\s+[\"|\'](?P<version>.+)[\"\']')
    abs_path = Path(_cwd / rel_path)
    for line in abs_path.open():
        match = pattern.match(line.strip())
        if match:
            return match['version']
    raise RuntimeError("Unable to find version string.")

def _get_requirements(rel_path):
    abs_path = Path(_cwd / rel_path)
    return [_.strip() for _ in abs_path.open().readlines()]

CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

LONG_DESCRIPTION = (
    "Orchard is a collection of generic implementation of models that support both tensor & pipeline parallel"
    " optimization. The ranked models can be generated using Olive (https://microsoft.github.io/Olive)."
)

DESCRIPTION = LONG_DESCRIPTION.split(".", maxsplit=1)[0] + "."

setup(
    name="orchard",
    version=_get_version("orchard/__init__.py"),
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Microsoft Corporation",
    author_email="olivedevteam@microsoft.com",
    license="MIT License",
    classifiers=CLASSIFIERS,
    url="https://microsoft.github.io/Olive/",   # FIXME!
    download_url="https://github.com/devang-ml/orchard/tags",
    packages=find_packages(include=["orchard*"]),
    python_requires=">=3.8.0",
    install_requires=_get_requirements("requirements.txt"),
    extras_require={},
    include_package_data=False,
    package_data={},
    data_files=[],
)
