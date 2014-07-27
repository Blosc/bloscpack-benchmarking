#!/usr/bin/env python

""" Award points for position. """

import benchmark_analysis_utils as bau
import pandas as pd
import sys



def aggregate(df, ratio=False):
    values = ['compress', 'decompress', 'dc_no_cache']
    if ratio:
        values.append('ratio')
    results = {}
    for size in ('small', 'mid', 'large'):
        for storage in ('ephemeral', 'esb'):
            for complexity in ('arange', 'linspace', 'poisson', 'neuronal', 'bitcoin'):
                for value in values:
                    it = df.loc[(size, storage, complexity)].sort(value,
                            ascending=value=='ratio')[value]
                    if ratio:
                        codecs = set(df.index.levels[-2]).difference(set(('tables', 'npy')))
                        it = it.loc[codecs]
                    for i,(index, value)in enumerate(it.iteritems(),start=1):
                        #print i, "_".join(map(str,index)), value
                        codec = "_".join(map(str,index))
                        if codec not in results:
                            results[codec] = i
                        else:
                            results[codec] += i
    return results

df = bau.load_results_file(sys.argv[1]).sort()
df_results = pd.DataFrame.from_dict(aggregate(df, ratio=True), orient='index').sort(0)
df_results.index.names = ('codec',)
df_results.columns = ('score',)
df_results.to_csv('aggregate_with_ratio.csv')

df = bau.load_results_file(sys.argv[1]).sort()
df_results = pd.DataFrame.from_dict(aggregate(df, ratio=False), orient='index').sort(0)
df_results.index.names = ('codec',)
df_results.columns = ('score',)
df_results.to_csv('aggregate_without_ratio.csv')
