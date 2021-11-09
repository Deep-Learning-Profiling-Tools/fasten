# Always prefer setuptools over distutils
from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension
import os

operators_dir = os.path.join('fasten', 'operators')


def make_cpp_extension(op):
    op_file = os.path.join(operators_dir, op + '.cc')
    op_name = op + '_cpp'
    return cpp_extension.CppExtension(op_name, [op_file], extra_compile_args=['-O3', '-g'])


def make_cuda_extension(op):
    op_file = os.path.join(operators_dir, op + '.cu')
    op_name = op + '_cuda'
    return cpp_extension.CUDAExtension(op_name, [op_file],
                                       extra_compile_args={'nvcc': ['-O3', '-g', '-lineinfo'],
                                                           'cxx': ['-O3', '-g']})


# Get the long description from the README file
with open('README.md', 'r') as f:
    long_description = f.read()

cpp_extensions = [make_cpp_extension('bmm')]
cuda_extensions = [make_cuda_extension('bmm')]
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

    packages=find_packages(),  # Required

    python_requires='>=3.6, <4',

    install_requires=['torch', 'torch_geometric', 'pytest'],  # Optional

    extras_require={  # Optional
        'test': ['pytest'],
    },

    ext_modules=cpp_extensions + cuda_extensions,

    cmdclass={'build_ext': cpp_extension.BuildExtension},

    project_urls={  # Optional
        'Source': 'https://github.com/Jokeren/fasten'
    },

    include_package_data=True,

    long_description=long_description,

    long_description_content_type='text/markdown'
)
