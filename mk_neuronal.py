#!/usr/bin/env python
import numpy as np
import bloscpack as bp

print 'loading data..'
z = np.load('prod-iaf-dst-cont-1000-ai-na-120.00m-1395826618.6076007.npz')
d = dict((z.items()))
d.pop('data_keys')
a = np.hstack((d.values()))

base = 2**(20)
for name, size in (('small', base / 8),
                    ('mid', base * 10 / 8),
                    ('large', base * 100 / 8),
                    ):
    print 'save: %s' % name
    np.save('neuronal_%s' % name, a[:size])
    bp.pack_ndarray_file(a[:size], 'neuronal_%s.blp' % name)


print 'make xlarge'
x = np.hstack([a, a, a, a, a])
size = base * 1000 /8
name = 'xlarge'
np.save('neuronal_%s' % name, x[:size])
bp.pack_ndarray_file(x[:size], 'neuronal_%s.blp' % name)
