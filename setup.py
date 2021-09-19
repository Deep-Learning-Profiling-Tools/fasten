# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='fasten',  # Required

    version='0.0.1',  # Required

    description='A libary for message passing based heterogenous graphs',  # Optional

    author='Keren Zhou',  # Optional

    author_email='keren.zhou@rice.edu',  # Optional

    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Pick your license as you wish
        'License :: OSI Approved :: BSD-2 License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],

    package_dir={'': 'python'},  # Optional

    packages=find_packages(where='python'),  # Required

    python_requires='>=3.6, <4',

    install_requires=['torch', 'torch_geometric'],  # Optional

    extras_require={  # Optional
        'test': ['coverage'],
    },

    project_urls={  # Optional
        'Source': 'https://github.com/Jokeren/fasten'
    },
)
