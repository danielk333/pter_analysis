apter
======

``apter`` is a complementary analysis tool to the todotxt handler ``pter``. If certain tags are consequently used in tasks several statistics and distributions can be calculated using ``apter`` such as:

* Estimated time left per project
* Estimated workload for completing all tasks in ``pter`` search(es)
* Task estimation accuracy
* Task delay (usage of the ``t:`` tag) or task completion time (``completed`` before or after ``due:`` tag) distributions

And much more...

Installation
-------------

.. code-block:: bash

    pip install pter_analysis


Examples
---------

Example image 1

![](doc/apter-demo-1.png)


Example image 2

![](doc/apter-demo-1.png)


Example image 3

![](doc/apter-demo-1.png)



Tags used
----------

The analysis is based on the usage of these tags:

* ``due:`` The due date of the task, see ``pter`` docs for relative dates.
* ``estimate:`` The estimated time needed to complete task in ``hm`` format, e.g. ``2h30m``, see ``pter`` docs.
* ``spent:`` The actual time spent on the task, see ``pter`` docs on task tracking.


To facilitate usage of these tags, its recommended that highligts are set for them, in the terminal-version of ``pter`` this can be done trough the configuration files ``[Highlight]`` section. An example on ubuntu-16 terminal colors are:


.. code-block:: 

    [Highlight]
    due = 33
    estimate = 207
    spent = 118
    t = 45


Configuration
---------------

``apter`` searches for a configuration file by default in a ``pter_analysis`` folder in the system default configuration folder, usually ``~/.config``. The configuration file is named ``pter_analysis.conf``.

Below is an example configuration file:


.. code-block:: 

    [General]
    work-day-length = 6h
    work-week-length = 5d
    todotxt-file = /path/to/my/tasklist.todotxt
    matplotlib-style = Solarize_Light2
    usetex = True
    search-case-sensitive = True


Each function call to ``apter`` can also include a custom configuration file using the ``--config [path]`` option.

Most of the above options are self explanatory. The ``usetex`` options enables latex formating of matplotlib if ``True`` and a list of standard ``matplotlib-style``'s can be found on the matplotlib docs <https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html>. A path to a custom style can also be supplied here.