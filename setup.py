import setuptools
__version__ = '0.3.0'

with open('README.rst', 'r') as fh:
    long_description = fh.read()


with open('requirements', 'r') as fh:
    pip_req = fh.read().split('\n')
    pip_req = [x.strip() for x in pip_req if len(x.strip()) > 0]


setuptools.setup(
    name='pter_analysis',
    version=__version__,
    long_description=long_description,
    url='https://github.com/danielk333/pter_analysis',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU-GPLv3',
        'Operating System :: OS Independent',
    ],
    install_requires=pip_req,
    packages=setuptools.find_packages(),
    entry_points={'console_scripts': ['apter=pter_analysis.main:cli']},
    # metadata to display on PyPI
    author='Daniel Kastinen',
    author_email='daniel.kastinen@irf.se',
    description='Analysis of todotxt files using pter',
    license='GNU-GPLv3',
)