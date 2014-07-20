#!/usr/bin/env python
import numpy as np
import pandas as pd
import bloscpack as bp

print 'loading data..'
df = pd.read_csv('mtgoxUSD.csv', names=['date', 'price', 'volume'])
a = np.array(df[['price', 'volume']]).T.copy()

base = 2**(20)
for name, size in (('small', base / 16),
                    ('mid', base * 10 / 16),
                    ('large', base * 100 /16),
                    ):
    print 'save: %s' % name
    np.save('bitcoin_%s' % name, a[:, :size])
    bp.pack_ndarray_file(a[:, :size], 'bitcoin_%s.blp' % name)
