# Always prefer setuptools over distutils
from setuptools import find_packages, setup

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
    install_requires=['pytest', 'flake8', 'autopep8', 'isort', 'pre-commit', 'pytest', 'pyg-nightly', 'pyg-lib@https://github.com/pyg-team/pyg-lib.git'],
    include_package_data=True,
    long_description_content_type='text/markdown'
)
