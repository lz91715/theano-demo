import numpy as np
import theano.tensor as T
from theano import function
from theano import In

# activation function example
x = T.dmatrix('x')
s = 1/(1+T.exp(-x))
logistic = function([x], s)

print logistic([[0,1],[-2,-3]])

# multiply outputs for a function
a,b = T.dmatrices('a','b')
diff = a - b
abs_diff = abs(diff)
diff_squared = abs_diff ** 2
f = function([a,b],[diff,abs_diff,diff_squared])
print f(np.ones((2,2)), np.arange(4).reshape((2,2)))

# name for a function
x,y,w = T.dscalars('x','y','w')
z = (x+y)*w
f = function([x,In(y, value = 1),In(w, value = 2, name = "weights")],z)

print f(23,)
