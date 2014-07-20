#!/usr/bin/env python

""" Award points for position. """

import benchmark_analysis_utils as bau
import sys

df = bau.load_results_file(sys.argv[1]).sort()

results = {}

for size in ('small', 'mid', 'large'):
    for storage in ('ephemeral', 'esb'):
        for complexity in ('arange', 'linspace', 'poisson', 'neuronal', 'bitcoin'):
            for value in ('compress', 'decompress', 'dc_no_cache'):
                it = df.loc[(size, storage, complexity)].sort(value, ascending=False)[value]
                for i,(index, value)in enumerate(it.iteritems(),start=1):
                    #print i, "_".join(map(str,index)), value
                    codec = "_".join(map(str,index))
                    if codec not in results:
                        results[codec] = i
                    else:
                        results[codec] += i


for i in sorted([(v,k) for k,v in results.items()]):
    print i


