#!/usr/bin/env python3

import numpy as np
from scipy import spatial
from scipy import stats

m1 = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0],
               [1, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0]])

m2 = np.array([[1, 1, 1, 1, 4, 1, 1, 1, 1],
               [1, 1, 1, 1, 4, 1, 1, 1, 1],
               [1, 1, 1, 1, 4, 1, 1, 1, 1],
               [1, 1, 1, 1, 4, 1, 1, 1, 1],
               [4, 4, 4, 4, 4, 4, 4, 4, 4],
               [1, 1, 1, 1, 4, 1, 1, 1, 1],
               [1, 1, 1, 1, 4, 1, 1, 1, 1],
               [1, 1, 1, 1, 4, 1, 1, 1, 1],
               [1, 1, 1, 1, 4, 1, 1, 1, 1]])

m3 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 1, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 1, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 1, 0],
               [1, 0, 0, 0, 0, 0, 0, 0, 1]])

m4 = np.random.randint(5, size=(9, 9))

for idx, m in enumerate((m1, m2, m3, m4)):
    print('m%d' % (idx + 1))
    print('- max: %d' % m.max())
    print('- mean: %d' % m.mean())
    print('- sum: %d' % m.sum())
    print('- std: %d' % np.std(m))
    print(stats.describe(m))

methods = {
    'braycurtis': spatial.distance.braycurtis,
    'canberra': spatial.distance.canberra,
    'chebyshev': spatial.distance.chebyshev,
    'cityblock': spatial.distance.cityblock,
    'correlation': spatial.distance.correlation,
    'cosine': spatial.distance.cosine,
    'euclidean': spatial.distance.euclidean,
    'hamming': spatial.distance.hamming,
    'jensenshannon': spatial.distance.jensenshannon,
    'minkowski': spatial.distance.minkowski,
    'sqeuclidean': spatial.distance.sqeuclidean
}

# euclidean distance
print('np.linalg.norm(m1 - m2): %f' % np.linalg.norm(m1 - m2))
print('np.linalg.norm(m1 - m3): %f' % np.linalg.norm(m1 - m3))
print('np.linalg.norm(m2 - m3): %f' % np.linalg.norm(m2 - m3))
print('np.linalg.norm(m1 - m4): %f' % np.linalg.norm(m1 - m4))
print('np.linalg.norm(m2 - m4): %f' % np.linalg.norm(m2 - m4))
print('np.linalg.norm(m3 - m4): %f' % np.linalg.norm(m3 - m4))

fm1 = m1.ravel()  # Flattened array
fm2 = m2.ravel()
fm3 = m3.ravel()
fm4 = m4.ravel()
for m in methods.keys():
    print('%s(m1, m2): %f' % (m, methods[m](fm1, fm2)))
    print('%s(m1, m3): %f' % (m, methods[m](fm1, fm3)))
    print('%s(m2, m3): %f' % (m, methods[m](fm2, fm3)))
    print('%s(m1, m4): %f' % (m, methods[m](fm1, fm4)))
    print('%s(m2, m4): %f' % (m, methods[m](fm2, fm4)))
    print('%s(m3, m4): %f' % (m, methods[m](fm3, fm4)))

_, _, d = spatial.procrustes(m1, m2)
print('spatial.procrustes(m1, m2): %f' % d)
_, _, d = spatial.procrustes(m1, m3)
print('spatial.procrustes(m1, m3): %f' % d)
_, _, d = spatial.procrustes(m2, m3)
print('spatial.procrustes(m2, m3): %f' % d)
_, _, d = spatial.procrustes(m1, m4)
print('spatial.procrustes(m1, m4): %f' % d)
_, _, d = spatial.procrustes(m2, m4)
print('spatial.procrustes(m2, m4): %f' % d)
_, _, d = spatial.procrustes(m3, m4)
print('spatial.procrustes(m3, m4): %f' % d)
