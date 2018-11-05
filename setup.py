#!/usr/bin/env python

import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "Xana",
    version = "0.0.8",
    author = "Mario Reiser",
    author_email = "mario.mkel@gmail.com",
    description = ("A data Analysis software for XPCS and XSVS experiments."),
#    license = "BSD",
    keywords = "analysis, XPCS, XSVS, XFEL",
    url = "",
    packages=[],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Data Analysis",
#        "License :: OSI Approved :: BSD License",
    ],
)
