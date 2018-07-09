
# coding: utf-8

# In[ ]:


"""
This project was developed by Peter Chen, Rocky Duan, Pieter Abbeel for the Berkeley Deep RL Bootcamp, August 2017.
Bootcamp website with slides and lecture videos: https://sites.google.com/view/deep-rl-bootcamp/.

Code adapted from Stanford CS231N materials: http://cs231n.stanford.edu/
"""


# note to properly run this lab, you should execute all code blocks sequentially
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from collections import namedtuple, defaultdict, deque

import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import matplotlib.pyplot as plt


# ## Introduction to Chainer
#
# Chainer can be understood as Numpy plus the ability to record the computation graph of numerical operations to enable Automatic Differentiation. (Chainer actually also offers many other things; for example, a Numpy equivalent library that runs on GPU, but we will ignore them for now)

# Let's illustrate how Chainer works by a simple 1D regression task.
#
# Suppose we have observations from the following model $y = w x + b + \epsilon$ where $\epsilon \sim \mathcal{N}(0, 0.1)$ and the task is to estimate the linear model parameters $w, b$ from data.

# In[3]:


# first generate some observations
true_a = 1.3
true_b = 0.4
data_x = (np.arange(100) / 99.0 - .5).astype(np.float32)  # Chainer assumes all the cpu computation is done in float32
data_y = (data_x * true_a + true_b + np.random.randn(*data_x.shape) * 0.1).astype(np.float32)
_ = plt.scatter(data_x, data_y, c='b')


# Chainer provides an abstraction called `Link` that describe some computation and keeps track of parameters for it. For instance, a `Linear` link describes a linear map on input and keeps track of `w` and bias `b`.

# In[4]:

# chianer.links.Linear --> L.Linear
# Linear layer (a.k.a. fully-connected layer)
# W = Weight parameter (Matrix)
# b = bias parameter (Vector)
# y = mx + c
# m = W ; b = c
model = L.Linear(in_size=1, out_size=1)  # input is 1D data and output is also 1D data

# Chainer will randomly initialize `w` and `b` for us.
# we can take a look at their values
print("w:", model.W)
print("b:", model.b)

# model.W and model.b have type `chainer.Variable`,
#   which is a wrapper around Numpy array
assert isinstance(model.W, chainer.Variable)

# operations that involve `chainer.Variable` will produce
#   `chainer.Variable` and this records the computation graph
var_result = model.W + 123  # some random computation
print("Operations on chainer.Variable: %s, type: %s" % (var_result, type(var_result)))

# the underlying numpy array can be accessed by `data` attribute
print("numpy arrays:", model.W.data, var_result.data)


# In[5]:


# A chainer link is a callable object. calling it performs the
#   forward computation. (in this case, it performs Wx + b)
model_y = model(data_x[:, None])  # chainer's link usually assumes input is [Batch Size, Input Dimension]
# `model_y` is a chainer variable so we use `.data` to access its numpy array for plotting

# we can plot the model's current fit in red. it should be terrible because we haven't trained it yet
_ = plt.plot(data_x, model_y.data[:, 0], c='r')
_ = plt.scatter(data_x, data_y, c='b')
_ = plt.title("Initial model")
###############
# USER OUTPUT
###############
_ = plt.grid()
plt.clf()
# _ = plt.show()

# In[11]:


# now let's walk through how to perform forward computation
#  and use AD to get gradients

# first we clear the gradients that are stored in the model
model.cleargrads()
# as we have seen we can perform forward computation by calling the link
model_y = model(data_x[:, None])

# remember that `model_y` is a chainer variable. to operate on chainer variable
#    we will use functions from chainer.functions to operate on those objects.
# F.square ==> y_i = (x_i)²
loss = F.mean(F.square(model_y - data_y[:, None]))
# `loss` is a scalar chainer variable
assert isinstance(loss, chainer.Variable)
print("loss", loss)
# calculating gradients d loss /d params is as simple as
loss.backward()

# we can inspect the gradient of loss with respect to W
print("dloss/dW", model.W.grad)


# Now that we know how to calculate gradients, we can code up a simple loop to perform gradient descent to train this model:
#
# (Hint: if you run into weird problems, maybe the state has been messed up and you can
# try re-runing all the code blocks from the beginning)

# In[19]:


# now we can perform gradient descent to improve this model
# L.Linear = chainer.links.Linear --> http://docs.chainer.org/en/stable/reference/generated/chainer.links.Linear.html
# Linear layer (a.k.a. fully-connected layer)
# weight matrix = W
# bias vector = b
# y = mx + c
# m = W ; b = c
model = L.Linear(in_size=1, out_size=1)
losses = []

for i in range(100):
    model.cleargrads()
    # F.square ==> y_i = (x_i)²
    # model: data_y = model(data_x)
    loss = F.mean(F.square(model(data_x[:, None]) - data_y[:, None]))
    losses.append(float(loss.data))
    # *** YOUR CODE HERE TO PERFORM GRADIENT DESCENT ***
    loss.backward()

    data_W_old = model.W.data[:]
    lrate = .90
    # .91 funny
    if losses[-1] <= loss.data:
        model.b.data[:] = model.b.data[:] + (1 - lrate) * loss.data
    
    elif losses[-1] > loss:
        model.b.data[:] = model.b.data[:] - (1 - lrate) * loss.data

    if model.W.grad <= 0:
        model.W.data[:] = model.W.data[:] + (1 - lrate) * data_W_old

    elif model.W.grad > 0:
        model.W.data[:] = model.W.data[:] - (1 - lrate) * data_W_old


    # Hint: you could access gradients with model.W.grad, model.b.grad
    # Hint2: you could write data into a parameter with model.W.data[:] = some_numpy_array
    # Hint3: if your model doesn't learn, remember to try different learning rates
    if i % 25 == 0:
        print("Itr", i, "loss:", loss)
plt.plot(np.array(losses))
plt.title("Learning curve")
plt.figure()
plt.plot(data_x, model(data_x[:, None])[:, 0].data, c='r')
plt.scatter(data_x, data_y, c='b')
_ = plt.title("Trained model fitness")

plt.grid()
plt.show()


# ## Train your first deep model
#
# Now we have learned the basics of Chainer. We can use it to train a deep model to classify MNIST digits. We will train a model on the MNIST dataset because the dataset is small.
#
# First we load the data and see what the images look like:

# In[6]:


train, test = chainer.datasets.get_mnist()
# use train[data_point_index] to access data
print("train[i][0] is the ith image that's flattened, and has shape:", train[12][0].shape)
print("train[i][1] is the ith image's label, such as:", train[12][1])
# here we visualize two of them
plt.imshow(train[12][0].reshape([28, 28, ]))
plt.title("Label: %s" % train[12][1])
plt.figure()
plt.imshow(train[42][0].reshape([28, 28, ]))
_ = plt.title("Label: %s" % train[42][1])


# Next we will provide some boilerplate code and train a linear classifier as an example:

# In[ ]:


def run(model, batchsize=16, num_epochs=2):

    optimizer = chainer.optimizers.Adam()  # we will use chainer's Adam implementation instead of writing our own gradient based optimization
    optimizer.setup(model)

    stats = defaultdict(lambda: deque(maxlen=25))
    for epoch in range(num_epochs):
        train_iter = chainer.iterators.SerialIterator(train, batchsize, repeat=False, shuffle=True)
        test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

        for itr, batch in enumerate(train_iter):
            xs = np.concatenate([datum[0][None, :] for datum in batch])
            ys = np.array([datum[1] for datum in batch])

            logits = model(xs)

            loss = F.softmax_cross_entropy(logits, ys)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            # calculate stats
            stats["loss"].append(float(loss.data))
            stats["accuracy"].append(float((logits.data.argmax(1) == ys).sum() / batchsize))
            if itr % 300 == 0:
                print("; ".join("%s: %s" % (k, np.mean(vs)) for k, vs in stats.items()))

# try a simple linear model
# TODO: uncomment
# run(L.Linear(None, 10))


# Next we will try to improve performance by training an MLP instead. A partial implementation is provided for you to fill in:

# In[ ]:


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        # *** YOUR CODE HERE TO BUILD AN MLP W/ self.l1, self.l2, self.l3 ***
        #
        # Hint: you should make use of non-linearities / activation functions
        #     https://docs.chainer.org/en/stable/reference/functions.html#activation-functions
        raise NotImplementedError()
# TODO: uncomment
# run(MLP(200, 10))


# Next you should try to implement logging test loss and see if the model is overfitting.

# In[ ]:


def better_run(model, batchsize=16, num_epochs=2):

    optimizer = chainer.optimizers.Adam()  # we will use chainer's Adam implementation instead of writing our own gradient based optimization
    optimizer.setup(model)

    stats = defaultdict(lambda: deque(maxlen=25))
    for epoch in range(num_epochs):
        train_iter = chainer.iterators.SerialIterator(train, batchsize, repeat=False, shuffle=True)
        for itr, batch in enumerate(train_iter):
            xs = np.concatenate([datum[0][None, :] for datum in batch])
            ys = np.array([datum[1] for datum in batch])

            logits = model(xs)

            loss = F.softmax_cross_entropy(logits, ys)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            # calculate stats
            stats["loss"].append(float(loss.data))
            stats["accuracy"].append(float((logits.data.argmax(1) == ys).sum() / batchsize))
            if itr % 300 == 0:
                test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)
                # *** YOUR CODE implement logging of stats on test set ***
                print("; ".join("%s: %s" % (k, np.mean(vs)) for k, vs in stats.items()))

# TODO: uncomment
# better_run(MLP(200, 10))


# Try different variants!
#
# - Does using a ConvNet improve performance (reduce overfitting?)
# - Try changing the learning rate and observe the effect
# - Does the model train if you give it correlated gradients? (consecutively sample many batches of "1", then many batches of "2", ... etc
