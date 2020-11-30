import setuptools
__version__ = '1.0.1'

with open('README.md', 'r') as fh:
    long_description = fh.read()


with open('requirements', 'r') as fh:
    pip_req = fh.read().split('\n')
    pip_req = [x.strip() for x in pip_req if len(x.strip()) > 0]


setuptools.setup(
    name='pter_analysis',
    version=__version__,
    long_description='''
``apter`` is a complementary analysis tool to the todotxt handler ``pter``. If certain tags are consequently used in tasks several statistics and distributions can be calculated using ``apter`` such as:

* Estimated time left per project
* Estimated workload for completing all tasks in ``pter`` search(es)
* Task estimation accuracy
* Task delay (usage of the ``t:`` tag) or task completion time (``completed`` before or after ``due:`` tag) distributions

And much more...

See the github README file for more information.
    ''',
    url='https://github.com/danielk333/pter_analysis',
    classifiers=[
        'Programming Language :: Python :: 3',
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