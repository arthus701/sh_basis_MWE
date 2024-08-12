import os
import codecs
import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext


# https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


name = 'sh_basis'
version = get_version('sh_basis/__init__.py')
description = '''Minmal working example for sh basis'''
copyright = '2024 Maximilian Schanner'

ext_modules = [
    Pybind11Extension("_csh_basis",
                      ["sh_basis/csrc/main.cpp"],
                      define_macros=[('VERSION_INFO', version)],
                      extra_compile_args=[
                        "-march=native",
                        "-O3",
                        "-fopenmp"
                      ],
                      libraries=["gomp"],
                      ),
]

setuptools.setup(
    name=name,
    version=version,
    author='Schanner, M. A.',
    author_email='arthus@gfz-potsdam.de',
    packages=['sh_basis'],
    license='GPL v3',
    description=description,
    long_description=description,
    install_requires=[
        'numpy>=1.18',
        'pyshtools',
        'pybind11',
        'tqdm',
        ],
    ext_modules=ext_modules,
    data_files=[('sh_basis', ['sh_basis/csrc/dspharm.h'])],
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
)
