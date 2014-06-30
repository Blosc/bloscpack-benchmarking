Large Scale Bloscpack Benchmarking Utilities
============================================

Some utilities for running and analysing Bloscpack benchmarks.

Derived from: https://github.com/esc/euroscipy2013-talk-bloscpack

Licence: MIT

Running the Benchmarks
----------------------

You need root access to run these benchmarks, since it will be dropping the
file system caches. If you are using Anconda, the following incantation using
``sudo`` might help:

.. code-block:: console

    $ sudo env PATH=$PATH PYTHONPATH=. python benchmark.py
