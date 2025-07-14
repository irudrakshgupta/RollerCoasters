#!/usr/bin/env python
#
# Copyright 2014 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function
import os
import re
import sys
from operator import lt, gt, eq, le, ge
from os.path import (
    abspath,
    dirname,
    join,
)
from distutils.version import StrictVersion
from setuptools import (
    Extension,
    find_packages,
    setup,
)

import versioneer


class LazyBuildExtCommandClass(dict):
    """
    Lazy command class that defers operations requiring Cython and numpy until
    they've actually been downloaded and installed by setup_requires.
    """
    def __contains__(self, key):
        return (
            key == 'build_ext'
            or super(LazyBuildExtCommandClass, self).__contains__(key)
        )

    def __setitem__(self, key, value):
        if key == 'build_ext':
            raise AssertionError("build_ext overridden!")
        super(LazyBuildExtCommandClass, self).__setitem__(key, value)

    def __getitem__(self, key):
        if key != 'build_ext':
            return super(LazyBuildExtCommandClass, self).__getitem__(key)

        from Cython.Distutils import build_ext as cython_build_ext
        import numpy

        # Cython_build_ext isn't a new-style class in Py2.
        class build_ext(cython_build_ext, object):
            """
            Custom build_ext command that lazily adds numpy's include_dir to
            extensions.
            """
            def build_extensions(self):
                """
                Lazily append numpy's include directory to Extension includes.

                This is done here rather than at module scope because setup.py
                may be run before numpy has been installed, in which case
                importing numpy and calling `numpy.get_include()` will fail.
                """
                numpy_incl = numpy.get_include()
                for ext in self.extensions:
                    ext.include_dirs.append(numpy_incl)

                super(build_ext, self).build_extensions()
        return build_ext


def window_specialization(typename):
    """Make an extension for an AdjustedArrayWindow specialization."""
    return Extension(
        'zipline.lib._{name}window'.format(name=typename),
        ['zipline/lib/_{name}window.pyx'.format(name=typename)],
        depends=['zipline/lib/_windowtemplate.pxi'],
    )


ext_modules = [
    Extension('zipline.assets._assets', ['zipline/assets/_assets.pyx']),
    Extension('zipline.assets.continuous_futures',
              ['zipline/assets/continuous_futures.pyx']),
    Extension('zipline.lib.adjustment', ['zipline/lib/adjustment.pyx']),
    Extension('zipline.lib._factorize', ['zipline/lib/_factorize.pyx']),
    window_specialization('float64'),
    window_specialization('int64'),
    window_specialization('int64'),
    window_specialization('uint8'),
    window_specialization('label'),
    Extension('zipline.lib.rank', ['zipline/lib/rank.pyx']),
    Extension('zipline.data._equities', ['zipline/data/_equities.pyx']),
    Extension('zipline.data._adjustments', ['zipline/data/_adjustments.pyx']),
    Extension('zipline._protocol', ['zipline/_protocol.pyx']),
    Extension(
        'zipline.finance._finance_ext',
        ['zipline/finance/_finance_ext.pyx'],
    ),
    Extension('zipline.gens.sim_engine', ['zipline/gens/sim_engine.pyx']),
    Extension(
        'zipline.data._minute_bar_internal',
        ['zipline/data/_minute_bar_internal.pyx']
    ),
    Extension(
        'zipline.data._resample',
        ['zipline/data/_resample.pyx']
    ),
    Extension(
        'zipline.pipeline.loaders.blaze._core',
        ['zipline/pipeline/loaders/blaze/_core.pyx'],
        depends=['zipline/lib/adjustment.pxd'],
    ),
    # New extensions for enhanced features
    Extension(
        'zipline.ml._ml_ext',
        ['zipline/ml/_ml_ext.pyx'],
    ),
    Extension(
        'zipline.risk._risk_ext',
        ['zipline/risk/_risk_ext.pyx'],
    ),
    Extension(
        'zipline.realtime._realtime_ext',
        ['zipline/realtime/_realtime_ext.pyx'],
    ),
]


STR_TO_CMP = {
    '<': lt,
    '<=': le,
    '=': eq,
    '==': eq,
    '>': gt,
    '>=': ge,
}

SYS_VERSION = '.'.join(list(map(str, sys.version_info[:3])))


def _filter_requirements(lines_iter, filter_names=None,
                         filter_sys_version=False):
    for line in lines_iter:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        match = REQ_PATTERN.match(line)
        if match is None:
            raise AssertionError("Could not parse requirement: %r" % line)

        name = match.group('name')
        if filter_names is not None and name not in filter_names:
            continue

        if filter_sys_version and match.group('pyspec'):
            pycomp, pyspec = match.group('pycomp', 'pyspec')
            comp = STR_TO_CMP[pycomp]
            pyver_spec = StrictVersion(pyspec)
            if comp(SYS_VERSION, pyver_spec):
                # pip install -r understands lines with ;python_version<'3.0',
                # but pip install -e does not.  Filter here, removing the
                # env marker.
                yield line.split(';')[0]
            continue

        yield line


REQ_PATTERN = re.compile(
    r"(?P<name>[^=<>;]+)((?P<comp>[<=>]{1,2})(?P<spec>[^;]+))?"
    r"(?:(;\W*python_version\W*(?P<pycomp>[<=>]{1,2})\W*"
    r"(?P<pyspec>[0-9.]+)))?\W*"
)


def _conda_format(req):
    def _sub(m):
        name = m.group('name').lower()
        if name == 'numpy':
            return 'numpy x.x'
        if name == 'tables':
            name = 'pytables'

        comp, spec = m.group('comp', 'spec')
        if comp and spec:
            formatted = '%s %s%s' % (name, comp, spec)
        else:
            formatted = name
        pycomp, pyspec = m.group('pycomp', 'pyspec')
        if pyspec:
            # Compare the two-digit string versions as ints.
            selector = ' # [int(py) %s int(%s)]' % (
                pycomp, ''.join(pyspec.split('.')[:2]).ljust(2, '0')
            )
            return formatted + selector

        return formatted

    return REQ_PATTERN.sub(_sub, req, 1)


def read_requirements(path,
                      filter_names=None,
                      conda_format=False):
    """Read requirements from a file.

    Parameters
    ----------
    path : str
        Path to the requirements file.
    filter_names : set[str], optional
        If provided, only include requirements for packages in this set.
    conda_format : bool, optional
        If True, format requirements for conda instead of pip.

    Returns
    -------
    requirements : list[str]
        List of requirement strings.
    """
    with open(path) as f:
        requirements = list(_filter_requirements(
            f,
            filter_names=filter_names,
            filter_sys_version=True,
        ))

    if conda_format:
        requirements = [_conda_format(req) for req in requirements]

    return requirements


def install_requires(conda_format=False):
    """Get the list of packages required for installation.

    Parameters
    ----------
    conda_format : bool, optional
        If True, format requirements for conda instead of pip.

    Returns
    -------
    requirements : list[str]
        List of requirement strings.
    """
    return read_requirements('etc/requirements.in', conda_format=conda_format)


def extras_requires(conda_format=False):
    """Get the list of packages required for development.

    Parameters
    ----------
    conda_format : bool, optional
        If True, format requirements for conda instead of pip.

    Returns
    -------
    requirements : list[str]
        List of requirement strings.
    """
    return read_requirements('etc/requirements-dev.in', conda_format=conda_format)


def setup_requirements(requirements_path, module_names,
                       conda_format=False):
    """Get the list of packages required for setup.

    Parameters
    ----------
    requirements_path : str
        Path to the requirements file.
    module_names : list[str]
        List of module names to include.
    conda_format : bool, optional
        If True, format requirements for conda instead of pip.

    Returns
    -------
    requirements : list[str]
        List of requirement strings.
    """
    return read_requirements(
        requirements_path,
        filter_names=set(module_names),
        conda_format=conda_format,
    )


if __name__ == '__main__':
    # Get the long description from the README file
    here = abspath(dirname(__file__))
    with open(join(here, 'README.rst')) as f:
        long_description = f.read()

    setup(
        name='zipline',
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        description='A Pythonic algorithmic trading library.',
        long_description=long_description,
        author='Quantopian Inc.',
        author_email='opensource@quantopian.com',
        url='https://www.zipline.io',
        packages=find_packages(),
        include_package_data=True,
        install_requires=install_requires(),
        extras_require={
            'dev': extras_requires(),
            'all': [
                # Core ML dependencies
                'scikit-learn>=1.0.0',
                'tensorflow>=2.8.0',
                'torch>=1.12.0',
                'xgboost>=1.5.0',
                'lightgbm>=3.3.0',
                
                # GPU acceleration
                'cupy-cuda11x>=10.0.0',
                'numba>=0.56.0',
                
                # Real-time processing
                'kafka-python>=2.0.0',
                'websockets>=10.0',
                'aiohttp>=3.8.0',
                'asyncio-mqtt>=0.11.0',
                
                # Advanced data sources
                'yfinance>=0.1.70',
                'alpha-vantage>=2.3.0',
                'polygon-api-client>=1.0.0',
                'iexfinance>=0.4.0',
                
                # Visualization
                'plotly>=5.0.0',
                'dash>=2.0.0',
                'bokeh>=2.4.0',
                'holoviews>=1.14.0',
                
                # Web APIs
                'fastapi>=0.75.0',
                'uvicorn>=0.17.0',
                'graphql-core>=3.2.0',
                'grpcio>=1.44.0',
                
                # Cloud & deployment
                'kubernetes>=18.0.0',
                'docker>=5.0.0',
                'boto3>=1.24.0',
                'google-cloud-storage>=2.0.0',
                
                # Advanced analytics
                'scipy>=1.8.0',
                'statsmodels>=0.13.0',
                'arch>=5.0.0',
                'pykalman>=0.9.5',
                
                # Risk management
                'pyfolio>=0.9.2',
                'empyrical>=0.5.5',
                'pyfin>=0.1.0',
                
                # Data quality
                'great-expectations>=0.15.0',
                'pandera>=0.8.0',
                'cerberus>=1.3.0',
                
                # Performance monitoring
                'prometheus-client>=0.12.0',
                'jaeger-client>=4.8.0',
                'opentelemetry-api>=1.10.0',
                
                # Development tools
                'jupyter>=1.0.0',
                'ipywidgets>=7.6.0',
                'voila>=0.3.0',
                'streamlit>=1.20.0',
            ]
        },
        setup_requires=setup_requirements(
            'etc/requirements-setup.in',
            ['Cython', 'numpy'],
        ),
        ext_modules=ext_modules,
        cmdclass=LazyBuildExtCommandClass(),
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Financial and Insurance Industry',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Topic :: Office/Business :: Financial :: Investment',
        ],
        python_requires='>=3.8',
        zip_safe=False,
    )
