# Always prefer setuptools over distutils
import os

from setuptools import find_packages, setup

ci_only = os.environ.get("CI_ONLY", False)
pkgs = ['pytest', 'flake8', 'autopep8', 'isort', 'pre-commit', 'pytest']
if not ci_only:
    pkgs += ['pyg-nightly', 'pyg-lib@git+https://github.com/pyg-team/pyg-lib.git'],

setup(
    name='fasten',  # Required
    version='0.1',  # Required
    description='A libary for fast segment operators',  # Optional

    author='Keren Zhou',  # Optional

    author_email='kerenzhou@outlook.com',  # Optional

    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD-3 License'
    ],

    packages=find_packages(),  # Required

    python_requires='>=3.6, <4',
    install_requires=pkgs,
    include_package_data=True,
    long_description_content_type='text/markdown'
)
