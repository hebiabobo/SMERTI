import numpy as np

attn_shape = (1, 4, 4)
a = np.triu(np.ones(attn_shape), k=1)
print(a)
