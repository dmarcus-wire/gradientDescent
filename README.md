# Gradient Descent

[!image](./images/output.png)

Optimization algorithms:
- engines that power neural networks
- enable them to patterns from data

Parameterized learning:
W - weight matrix
b - bias vector

Gradient descent:
- iteratively evaluate parameters 
- iterative optimization algorithm that operators over a loss landscape
- compute loss
- make a small step in the direction to minimize loss
- 'vanilla' gradient descent
- 'stochastic' gradient descent
- 'analytic' gradient descent  
- local maxima - each peak in the plot
- local minimum - small regions of loss
- global minimum - local minimum with smallest region of loss

Bias trick:
- combining W (weight) mmatrix and b (bias vector) into a single parameter
- this adds an extra dim to the X (inout) data
- this is a constant, 1
- b is such embedded into the W
- enables learning a single matrix of weights

# Vanilla gradient descent
```angular2html
# loop until some condition is met
# 1. set number of epochs passed
# 2. loss has sufficient low 
# 2. training accurancy satisfactory high
# 3. Loss has not improved M subsequent epochs
while True:
    Wgradient = evaluate_gradient(loss, data, W)
    #  if alpha is too large, never descend
    # if alpha is too small, prohibitively many iteratiocs to implement
    W += -alpa * Wgradient
```