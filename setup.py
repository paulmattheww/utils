#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from ast import literal_eval
import os

<<<<<<< HEAD
#DOCKER_DEV = literal_eval(os.environ.get('DOCKER_DEV', '0'))
=======
DOCKER_DEV = literal_eval(os.environ.get('DOCKER_DEV', '0'))
>>>>>>> develop

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

<<<<<<< HEAD
requirements = ['atomicwrites', 'hashfile', ]
=======
requirements = ['atomicwrites', 'hashfile', 'pandas', 'numpy',
'matplotlib', 'seaborn', 'sklearn' ]
>>>>>>> develop

setup_requirements = ['pytest-runner', 'setuptools_scm', ]

test_requirements = ['pytest', ]

setup(
    author="Paul M. Washburn",
<<<<<<< HEAD
    author_email='paulmattheww@gmail.com',
=======
    author_email='paulmattheww',
>>>>>>> develop
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
<<<<<<< HEAD
    description="Reusable code collected over the years.",
=======
    description="Package of utilities from HES.",
>>>>>>> develop
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='utils',
    name='utils',
<<<<<<< HEAD
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/paulmattheww/utils',
    version='0.1.0',
    zip_safe=False,
    use_scm_version=1,
=======
    #packages=find_packages(include=['pset_utils']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    #url='https://github.com/Pset Utils/pset_utils',
    version='0.1.0',
    zip_safe=False,
    use_scm_version=not DOCKER_DEV,
>>>>>>> develop
)
