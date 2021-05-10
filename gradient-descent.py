# USAGE
# python gradient-descent.py

# import packages
# we want the model the generalize or predict on unseen data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# dummy dataset, 2D in separate blobs
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

# compute the sigmoid activation value for a given input
def sig_act(x):
    return 1.0 / (1 + np.exp(-x))

# compute the derivative of the sigmoid activation ASSUMING
# the input 'x' has already been passed through the sig_act function
# this will derive the actual gradient
def sig_der(x):
    return x * (1 - x)

# apply the sigmoid activation function and then threshold is based on
# whether the neuron is firing 1 or 0. W is what the model has learned
def predict(X, W):
    # takes the dot product between features and weights matrix
    preds = sig_act(X.dot(W))

    # applies a step function to threshold the outputs to binary class labels
    # if predicted value is <= 0.5, clamp the value
    preds[preds <= 0.5] = 0
    # if the predicted value is > 0, its also greater than 0.5, thus 1
    preds[preds > 0] = 1

    # return the predictions
    return preds

# given a set of input data points X and weights W, we call the sigmoid
# activation f(x) on them to obtain a set of predictions, we threshold the
# predictions = any prediction <= 0.5 is set to 0, any > 0.5 is set to 1

ap = argparse.ArgumentParser()
# the number of epochs we'll use to train our model using gradient descent
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
# learning rate is a hyperparameter to tune, typically set to 0.1, 0.01, 0.001 as initial values
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
args = vars(ap.parse_args())

# generate 1,000 datapoints separated into 2 classes
# each data point is a 2D feature vector implying features are length =2
# labels are either 0 or 1 and our goal is predict accurately
# 2 blobs of data
# 2 features
# 2 centers od the blobs
# 1.5 std away from center
# X = 2D datapoints
# y = 0 or 1
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# insert a NEW columns of 1's as an entry in the feature matrix
# bias trick is a set of constant values across all feature vectors allowing
# us to treat bias as a trainable parameter within W matrix instead of separtely
X = np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing splits using 50%
# of data for training and rest for testing
# trainX = training data
# testX = testing data
# trainY = train labels 0 or 1
# testY = test labels 0 or 1
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

# randomly initialize W matrix using uniform distribution so the same number of
# dimensions as input features (including bias)
# # of rows == # of columns
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses= []

# good initialization is critical to trianing a neural network in reasonable time

# initialize a list to keep track of losses after each epoch
# plot the loss to hopefully decrease over time
# start training and descent procedure

# loop over desired number of epochs
for epoch in np.arange(0, args["epochs"]):
    # take the dot product between feature X and W
    # pass the value through sigmoid function
    # result is prediction on entire dataset
    preds = sig_act(trainX.dot(W))

    # now we have prediction, determine the error
    # difference between prediction and true value
    # each epoch, each update of W, smooth loss minimization
    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)

    # gradient descent update is the dot product between
    # (1) features and (2) error of the sigmoid derivative
    d = error * sig_der(preds)
    # training data.Transpose.Dot product to measure error.
    gradient = trainX.T.dot(d)

    # this is the most critical step
    # in the update stage, "nudge" the weight
    # in the negative direction of the gradient
    # taking small steps to a more optimal parameter
    # weight matrix is updated by adding in the negative alpha
    # value (learning rate and ) times the gradient
    W += -args["alpha"] * gradient

    # check if an update should be displayed
    if epoch == 0 or (epoch +1 ) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1),
            loss))

# model is trained
# now evaluate
# to actually predictusing W matrix, call the predict method on testX and W
print("[INFO]  evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# visualize the output
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:, 0], s=30)

# construct the figure that plots loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epochs #")
plt.ylabel("Loss")
plt.show()

## Vanilla gradient descent only performs a weight update once for
# every epoch — in this example, we trained our model for 100 epochs,
# so only 100 updates took place.Depending on the initialization of the
# weight matrix and the size of the learning rate, it’s possible that we
# may not be able to learn a model that can separate the points (even though
# they are linearly separable).
#
# For simple gradient descent, you are better off training for more epochs
# with a smaller learning rate to help overcome this issue.
#
# However, a variant of gradient descent called Stochastic Gradient Descent performs
# a weight update for every batch of training data, implying there are multiple
# weight updates per epoch. This approach leads to a faster, more stable convergence.