from reorg import *
import cupy as xp
from chainer import Variable

x_data = xp.random.randn(100, 3, 32, 32).astype(xp.float32)
x = Variable(x_data)

y = reorg(x)
print(x.shape, y.shape)
