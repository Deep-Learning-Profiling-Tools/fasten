# Always prefer setuptools over distutils
from setuptools import find_packages, setup

setup(
    name='fasten',  # Required
    version='0.0.1',  # Required
    description='A libary for fast segment operators',  # Optional

    author='Keren Zhou',  # Optional

    author_email='kerenzhou@outlook.com',  # Optional

    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Pick your license as you wish
        'License :: OSI Approved :: BSD-2 License'
    ],

    packages=find_packages(),  # Required

    python_requires='>=3.6, <4',

    install_requires=['pytest',
                      'flake8',
                      'autopep8',
                      'isort',
                      'pre-commit'],  # Optional

    extras_require={  # Optional
        'test': ['pytest'],
    },

    project_urls={  # Optional
        'Source': 'https://github.com/Jokeren/fasten'
    },

    include_package_data=True,
    long_description_content_type='text/markdown'
)
