#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

version = {}
with open("deeptangle/version.py") as f:
    exec(f.read(), version)

setup(
    name="deeptangle",
    version=version["__version__"],
    license="MIT",
    author_email="albert.alonso@nbi.ku.dk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages()
)
