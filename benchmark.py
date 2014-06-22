#!/usr/bin/env python
# encoding: utf-8

""" Benchmarking utilities for comparing different Numpy array serialization
tools. """

import abc
import atexit
import gc
import os
from time import time
from collections import OrderedDict as od
import random

import progressbar as pbar
import numpy as np
from numpy.random import randn, poisson
import pandas as pd
import bloscpack as bp
import bloscpack.sysutil as bps
import joblib as jb
import sh
import yaml
import tables


def noop():
    pass


def gen_results_filename():
    return 'results_' + str(int(time()))


def extract_config():

    def git_sha(base=''):
        try:
            return str(sh.git('rev-parse', 'HEAD', _cwd=base)).strip()
        except Exception:
            return 'NA'

    config = c = {}

    versions = v = {}
    v['bloscpack'] = bp.__version__
    v['numpy']     = np.__version__
    v['joblib']    = jb.__version__
    v['conda']     = str(sh.conda('--version', _tty_in=True)).strip()
    v['python']     = str(sh.python('--version', _tty_in=True)).strip()

    hashes = h = {}
    h['bloscpack'] = git_sha(os.path.dirname(bp.__file__))
    h['joblib'] = git_sha(jb.__path__[0])
    h['numpy'] = git_sha(np.__path__[0])
    h['benchmark'] = git_sha()

    c['uname'] = str(sh.uname('-a')).strip()
    c['hostname'] = str(sh.hostname()).strip()
    c['whoami'] = str(sh.whoami()).strip()
    c['date'] = str(sh.date()).strip()

    c['versions'] = versions
    c['hashes'] = hashes
    return config

def vtimeit(stmt, setup=noop, before=noop, after=noop, repeat=3, number=3):
    """ Specialised version of the timeit utility.

    Supports special operations. `setup` is performed once before each set.
    `before` is performed before each run but IS NOT included in the timing.
    `after` is performed after each run and IS included in the timing. Garbage
    collection is disables during runs.

    A `run` is a single a execution of the code to be benchmarked. A set is
    collection of runs. Usually, one perform a `repeat` of sets with `number`
    number of runs. Then, the average across runs is taken for each set and the
    minimum is selected as the final timing value.

    Parameters
    ----------

    stmt : callable
        the thing to benchmark
    setup : callable
        callable to be executed once before all runs
    before : callable
        callable to be executed once before every run
    after : callable
        callable to be executed once after every run
    repeat : int
        the number of sets
    number : int
        the number of runs in each set

    Returns
    -------
    results : ndarray
        2D array with `repeat` rows and `number` columns.
    """

    result = np.empty((repeat, number))
    setup()
    for i in range(repeat):
        for j in range(number):
            before()

            gcold = gc.isenabled()
            gc.disable()

            tic = time()
            stmt()
            after()
            toc = time()
            result[i, j] = toc - tic

            if gcold:
                gc.enable()

    return result


def drop_caches():
    """ Drop linux file system caches. """
    bps.drop_caches()


def sync():
    """ Sync the linux file system buffers. """
    os.system('sync')


def make_arange_dataset(size):
    """ Make the dataset using arange"""
    return np.arange(size)


def make_linspace_dataset(size):
    return np.linspace(0, 1, size)


def make_sin_dataset(size):
    """ Make the dataset with medium entropy. """
    x = np.linspace(0, np.pi*2, 1e3)
    x = np.tile(x, size / len(x))
    assert len(x) == size
    y = np.sin(x)
    del x
    noise = np.random.randint(-1, 1, size) / 1e8
    it = y + noise
    del y
    del noise
    return it


def make_gaussian_dataset(size):
    """ Make a bunch of zero mean unit variance gaussian random numbers. """
    return randn(size)


def make_poisson_dataset(size):
    return poisson(5, size)


def reduce(result):
    """ Reduce the results array from a benchmark into a single value. """
    return result.mean(axis=1).min()


class AbstractRunner(object):
    """ Base class for a codec benchmark. """

    _metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compress(self):
        """ Implement this to benchmark compression. """
        pass

    @abc.abstractmethod
    def decompress(self):
        """ Implement this to benchmark decompression. """
        pass

    def clean(self):
        """ Delete output, sync the buffers and clean the cache. """
        if os.path.isfile(self.storage):
            os.remove(self.storage)
        sync()
        drop_caches()

    def deconfigure(self):
        """ Delete any storage internal to the instantiated object. """
        del self.ndarray
        gc.collect()

    def configure(self, ndarray, storage, level):
        """ Setup the input data, configure output and compression level. """
        self.ndarray = ndarray
        self.storage = os.path.join(storage, self.filename)
        self.level = level

    def ratio(self):
        """ Compute compression ratio. """
        return (float(os.path.getsize(self.storage)) /
                (self.ndarray.size * self.ndarray.dtype.itemsize))


class BloscpackRunner(AbstractRunner):
    """ Runner for Bloscpack.

    Files are generated without checksums, offsets and no preallocated chunks.
    """

    def __init__(self):
        self.name = 'bloscpack'
        self.bloscpack_args = bp.BloscpackArgs(offsets=False,
                                               checksum='None',
                                               max_app_chunks=0)
        self.filename = 'array.blp'

    def compress(self):
        blosc_args = bp.BloscArgs(clevel=self.level)
        bp.pack_ndarray_file(self.ndarray,
                             self.storage,
                             blosc_args=blosc_args,
                             bloscpack_args=self.bloscpack_args)

    def decompress(self):
        it = bp.unpack_ndarray_file(self.storage)


class NPZRunner(AbstractRunner):
    """ Runner for NPZ. """

    def __init__(self):
        self.name = 'npz'
        self.filename = 'array.npz'

    def compress(self):
        np.savez_compressed(self.storage, self.ndarray)

    def decompress(self):
        it = np.load(self.storage)['arr_0']


class NPYRunner(AbstractRunner):
    """ Runner for NPY. """

    def __init__(self):
        self.name = 'npy'
        self.filename = 'array.npy'

    def compress(self):
        np.save(self.storage, self.ndarray)

    def decompress(self):
        it = np.load(self.storage)


class ZFileRunner(AbstractRunner):
    """ Runner for ZFile. """

    def __init__(self):
        self.name = 'npy'
        self.filename = 'array.npy'

    @property
    def storage_size(self):
        return os.path.getsize(self.storage) + os.path.getsize(self.real_data)

    @property
    def real_data(self):
        return self.storage + '_01.npy.z'

    def clean(self):
        if os.path.isfile(self.storage):
            os.remove(self.storage)
        if os.path.isfile(self.real_data):
            os.remove(self.real_data)
        sync()
        drop_caches()

    def compress(self):
        jb.dump(self.ndarray, self.storage, compress=self.level, cache_size=0)

    def decompress(self):
        it = jb.load(self.storage)

    def ratio(self):
        return (float(self.storage_size) /
                (self.ndarray.size * self.ndarray.dtype.itemsize))


class PyTablesRunner(AbstractRunner):

    def __init__(self):
        self.name = 'pytables'
        self.filename = 'array.hdf5'

    def compress(self):
        f = tables.open_file(self.storage, 'w')
        if self.level != 0:
            filters = tables.Filters(complevel=self.level,
                                     complib='blosc',
                                     fletcher32=False)
        else:
            filters = None
        f.create_carray('/', 'array', obj=self.ndarray, filters=filters)
        f.close()

    def decompress(self):
        f = tables.open_file(self.storage)
        n = f.get_node('/', 'array')
        it = n.read()
        f.close()

if __name__ == '__main__':

    print "----------------------------------"
    success = False

    # handle configuration
    result_file_name = gen_results_filename()
    conf = yaml.dump(extract_config(), default_flow_style=False)
    print conf
    conf_file = result_file_name + '.info.yaml'
    with open(conf_file, 'w') as fp:
        fp.write(conf)
    print 'config saved to: ' + conf_file
    print "----------------------------------"

    # set up experimental parameters
    dataset_sizes = od([('small', 131072),
                        ('mid', 1310720),
                        #('large', 2e8),
                        ])
    storage_types = od([('ssd', '/tmp/bench'),
                        #('sd', '/mnt/sd/bench'),
                        ])
    complexity_types = od([('arange', make_arange_dataset),
                           ('linspace', make_linspace_dataset),
                           ('poisson', make_poisson_dataset),
                           #('medium', make_complex_dataset),
                           #('high', make_random_dataset),
                           ])
    codecs = od([('bloscpack', BloscpackRunner()),
                 ('tables', PyTablesRunner()),
                 ('npz', NPZRunner()),
                 ('npy', NPYRunner()),
                 ('zfile', ZFileRunner()),
                 ])

    codec_levels = od([('bloscpack', [1, 3, 7, 9]),
                       ('tables', [0, 1, 3, 7, 9]),
                       ('npz', [1, ]),
                       ('npy', [0, ]),
                       ('zfile', [1, 3, 7]),
                       ])

    for name, location in storage_types.items():
        if not os.path.isdir(location):
            raise Exception("Path %s at: '%s' does not exist!" % (name, location))

    columns = ['size',
               'storage',
               'complexity',
               'codec',
               'level',
               'compress',
               'decompress',
               'dc_no_cache',
               'ratio',
               ]

    sets = []
    # can't use itertools.product, because level depends on codec
    for size in dataset_sizes:
        for type_ in storage_types:
            for complexity in complexity_types:
                for codec in codecs:
                    for level in codec_levels[codec]:
                        sets.append((size, type_, complexity, codec, level))

    # shuffle the sets, so that
    random.shuffle(sets)

    # print configuration
    for name, config_dict in [('dataset_sizes', dataset_sizes),
                              ('storage_types', storage_types),
                              ('complexity_types', complexity_types),
                              ('codecs', codecs),
                              ('codec_levels', codec_levels),
                              ]:
            print name
            for k,v in config_dict.items():
                print "    ","%-20s %-20s" % (k,v)

    print "Total number of configurations: %d" % len(sets)
    print "----------------------------------"


    # setup output DataFrame
    n = len(sets)
    column_values = od(zip(columns, zip(*sets)))
    column_values['compress'] = np.zeros(n)
    column_values['decompress'] = np.zeros(n)
    column_values['dc_no_cache'] = np.zeros(n)
    column_values['ratio'] = np.zeros(n)

    results = pd.DataFrame(column_values)

    # define an atexit handler in case something goes wrong
    def temp_result():
        if not success:
            result_csv = 'TEMPORARY_RESULTS.csv'
            results.to_csv(result_csv)
            print 'ABORT: results acquired sofar saved to: ' + result_csv
    atexit.register(temp_result)

    # setup the progressbar
    class Display(pbar.Widget):
        """Displays the current count."""

        def update(self, pbar):
            try:
                return str(sets[pbar.currval])
            except IndexError:
                return ''
    widgets = ['Benchmark: ',
               pbar.Percentage(),
               ' ',
               pbar.Counter(), '/', str(n),
               ' ',
               Display(),
               ' ',
               pbar.Bar(marker='-'),
               ' ',
               pbar.AdaptiveETA(),
               ' ',
               ]
    pbar = pbar.ProgressBar(widgets=widgets, maxval=n).start()

    # go johnny go, go!
    for i, it in enumerate(sets):
        size, storage, complexity, codec, level = it

        if size == 'small':
            number = 10
            repeat = 10
        elif size == 'mid':
            number = 5
            repeat = 5
        elif size == 'large':
            number = 3
            repeat = 3
        else:
            raise RuntimeError("No such size: '%s'" % size)

        codec = codecs[codec]
        codec.configure(complexity_types[complexity](dataset_sizes[size]),
                        storage_types[storage], level)

        results['compress'][i] = reduce(vtimeit(codec.compress,
                                        setup=codec.compress,
                                        before=codec.clean, after=sync,
                                        number=number, repeat=repeat))
        results['ratio'][i] = codec.ratio()
        codec.deconfigure()
        results['decompress'][i] = reduce(vtimeit(codec.decompress,
                                                  setup=codec.decompress,
                                                  number=number,
                                                  repeat=repeat))
        results['dc_no_cache'][i] = reduce(vtimeit(codec.decompress,
                                                   before=drop_caches,
                                                   number=number,
                                                   repeat=repeat))

        codec.clean()
        pbar.update(i)

    pbar.finish()
    success = True
    result_csv = result_file_name + '.csv'
    results.to_csv(result_csv, index_label='id')
    print 'results saved to: ' + result_csv
