import pylab as plt

def result_filter(storage, size, codecs):
    # consider only ssd storage
    it = df[df['storage'] == storage]
    # consider only large test size
    it = it[it['size'] == size]
    # compare bloscpack with npy
    it = it[it['codec'].isin(codecs)]
    idx = list(it.columns)
    idx.pop(1) # remove the storage column
    idx.pop(-1) # remove the ratio column
    it = it[idx]
    # reindex using entropy, codec and level
    it = it.set_index(['entropy', 'codec', 'level'])
    return it

def plot_codec_comp( storage, codecs):
    it_small = result_filter(storage, 'small', codecs)
    it_medium = result_filter(storage, 'mid', codecs)
    it_large = result_filter(storage, 'large', codecs)
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15,15), sharey=True)
    for i, ent in enumerate(['low', 'medium', 'high']):
        it_small.ix[ent].plot(kind='barh', ax=ax[i, 0], legend=None)
        ax[i, 0].set_title('small size - %s entropy' % ent)
        it_medium.ix[ent].plot(kind='barh', ax=ax[i, 1], legend=None)
        ax[i, 1].set_title('medium size - %s entropy' % ent)
        it_large.ix[ent].plot(kind='barh', ax=ax[i, 2], legend=None)
        ax[i, 2].set_title('large size - %s entropy' % ent)

def plot_ratio(size, entropy, storage='ssd', ax=None):
    it = df
    it = it[it['storage'] == 'ssd']
    it = it[it['size'] == size]
    it = it[it['entropy'] == entropy]
    it = it[['level', 'codec', 'ratio']]
    it = it.set_index(['codec', 'level'])
    it.plot(kind='barh', figsize=(15, 8), ax=ax, legend=False,
            title="size: '%s' entropy: '%s'" % (size, entropy))

