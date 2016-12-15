import numpy as np
import theano
import theano.tensor as T
import pickle

# compute accuracy
def compute_accuracy(y_target, y_predict):
	correct_prediction = np.equal(y_target, y_predict)
	accuracy = 1.0*np.sum(correct_prediction)/len(correct_prediction)
	return accuracy

# fake data

N = 400			# training sample size
feats = 784		# number of input variables

D = (np.random.randn(N, feats), np.random.randint(size = N, low = 0, high = 2))

# declare variables
x = T.dmatrix('x')
y = T.dvector('y')

# construct graph
W = theano.shared(np.random.randn(feats), name = 'W')
b = theano.shared(0.1, name = 'b')
Wx_plus_b = T.dot(x, W) + b
activation_func = T.nnet.sigmoid(Wx_plus_b)
ans = activation_func > 0.5
cross_thropy = -y * T.log(activation_func) - (1-y) * T.log((1-activation_func))
# cost = cross_thropy.mean()
cost = cross_thropy.mean() + 0.1 * (W ** 2).sum() # L2 regularization
gW, gb = T.grad(cost, [W, b])

# compile
training_rate = 0.1
train = theano.function(inputs = [x, y],
						outputs = [ans, cost],
						updates = [(W, W - gW * training_rate),
									(b, b - gb * training_rate)])

predict = theano.function(inputs = [x], outputs = ans)
# training
for i in xrange(1000):
	pre, err = train(D[0], D[1])
	if i % 50 == 0:
		print err
		print compute_accuracy(D[1], predict(D[0]))

# save model 
with open('./model/classification_model.pickle','wb') as file:
	model = [W.get_value(), b.get_value()]
	pickle.dump(model, file)
	print W.get_value()[:10]

# load model
with open('./model/classification_model.pickle','rb') as file:
	model = pickle.load(file)
	W.set_value(model[0])
	b.set_value(model[1])
	print W.get_value()[:10]
