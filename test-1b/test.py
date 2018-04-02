import numpy as np
np.set_printoptions(threshold=np.nan)

import tensorflow as tf

b_size = 128

Ys = np.tile(np.expand_dims(np.arange(10), 1), [b_size, 1]).reshape(-1)
Ys_hot = np.zeros([b_size * 10, 10], np.float32)
Ys_hot[np.arange(b_size * 10), Ys] = 1

print(Ys_hot)
print(np.shape(Ys_hot))