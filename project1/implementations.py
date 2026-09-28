import numpy as np
from helpers.py import batch_iter

#TODO: implement logistic_regression and reg_logistic_regression


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """

    n = y.shape[0]
    weights = np.linalg.solve(tx.T @ tx, tx.T @ y)
    error = y - tx @ weights
    mse = (0.5/n) * (error.T @ error)

    return float(mse), weights

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    n = y.shape[0]
    weights = np.linalg.solve(tx.T @ tx + (2 * n * lambda_) * np.identity(tx.shape[1]), tx.T @ y)
    error = y - tx @ weights
    mse = (0.5/n) * (error.T @ error)

    return float(mse), weights

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    B = y.shape[0]
    e = y - tx@w
    return (-1/B) * np.transpose(tx) @ e


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    weights = [initial_w]
    w = initial_w

    for (minibatch_y, minibatch_tx) in enumerate(batch_iter(y, tx, batch_size=1, num_batches=max_iters)):
        grad = compute_stoch_gradient(minibatch_y, minibatch_tx, w)

        # Update model parameters
        w = w - gamma * grad
        # Compute the full-dataset loss (or mini-batch loss, depending on your lab convention)

        weights.append(w)

    n = np.shape(y)[0] 
    error = y - tx @ weights
    mse = (0.5/n) * (error.T @ error)

    return float(mse), weights

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    n = y.shape[0]
    e = y - tx@w
    return (-1/n) * np.transpose(tx) @ e

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters + 1 containing the model parameters as numpy arrays of shape (2, ),
            for each iteration of GD (as well as the final weights)
    """
    # Define parameters to store w and loss
    weights = [initial_w]
    w = initial_w
    for i in range(max_iters):

        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient

        # store w and loss
        weights.append(w)

    n = np.shape(y)[0] 
    error = y - tx @ weights
    mse = (0.5/n) * (error.T @ error)

    return float(mse), weights




