import numpy as np
import pytest

# Replace 'implementations' with the actual name of your Python file if different
from implementations import (
    least_squares,
    ridge_regression,
    compute_stoch_gradient,
    mean_squared_error_sgd,
    compute_gradient,
    mean_squared_error_gd,
    sigmoid,
    calculate_loss,
    calculate_gradient,
    logistic_regression,
    reg_logistic_regression,
)

# ---------------------------------------------------------------------------
# Fixtures for Reusable Test Data
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_data():
    """Simple dataset for linear regression and gradient descent tests."""
    y = np.array([0.1, 0.2, 0.3])
    tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.5, 2.0]])
    initial_w = np.array([0.5, 1.0])
    return y, tx, initial_w


@pytest.fixture
def logistic_data():
    """Simple binary dataset for logistic regression tests."""
    y = np.array([0.0, 1.0, 1.0, 0.0])
    tx = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [0.5, 1.0]])
    initial_w = np.array([0.1, -0.2])
    return y, tx, initial_w


# ---------------------------------------------------------------------------
# 1. Least Squares & Ridge Regression Tests
# ---------------------------------------------------------------------------


def test_least_squares_docstring():
    y = np.array([0.1, 0.2])
    tx = np.array([[2.3, 3.2], [1.0, 0.1]])
    expected_w = np.array([0.21212121, -0.12121212])
    expected_loss = 8.666684749742561e-33

    w, loss = least_squares(y, tx)

    assert w.shape == (2,)
    assert np.isscalar(loss) or loss.ndim == 0
    np.testing.assert_allclose(w, expected_w, rtol=1e-6)
    np.testing.assert_allclose(loss, expected_loss, atol=1e-15)


def test_ridge_regression_docstring():
    y = np.array([0.1, 0.2])
    tx = np.array([[2.3, 3.2], [1.0, 0.1]])

    # Lambda = 0 should match ordinary least squares
    w0, loss0 = ridge_regression(y, tx, 0)
    np.testing.assert_allclose(
        w0, np.array([0.21212121, -0.12121212]), rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(loss0, 0.0, atol=1e-15)

    # Lambda = 1
    w1, loss1 = ridge_regression(y, tx, 1)
    np.testing.assert_allclose(
        w1, np.array([0.03947092, 0.00319628]), rtol=1e-6, atol=1e-8
    )
    assert np.isscalar(loss1) or loss1.ndim == 0


# ---------------------------------------------------------------------------
# 2. Gradient Descent (GD) & Stochastic Gradient Descent (SGD) Tests
# ---------------------------------------------------------------------------


def test_compute_gradient(linear_data):
    y, tx, w = linear_data
    grad = compute_gradient(y, tx, w)

    # Manual check: e = y - tx @ w, grad = (-1/N) * tx.T @ e
    expected_e = y - tx @ w
    expected_grad = (-1.0 / len(y)) * (tx.T @ expected_e)

    assert grad.shape == w.shape
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-7)


def test_compute_stoch_gradient(linear_data):
    y, tx, w = linear_data
    grad = compute_stoch_gradient(y[:2], tx[:2], w)
    expected_grad = compute_gradient(y[:2], tx[:2], w)

    assert grad.shape == w.shape
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-7)


def test_mean_squared_error_gd_zero_iters(linear_data):
    y, tx, initial_w = linear_data
    w, loss = mean_squared_error_gd(y, tx, initial_w, max_iters=0, gamma=0.1)

    expected_error = y - tx @ initial_w
    expected_loss = (0.5 / len(y)) * (expected_error.T @ expected_error)

    np.testing.assert_allclose(w, initial_w)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-7)


def test_mean_squared_error_gd_step(linear_data):
    y, tx, initial_w = linear_data
    gamma = 0.05
    w, loss = mean_squared_error_gd(y, tx, initial_w, max_iters=1, gamma=gamma)

    expected_w = initial_w - gamma * compute_gradient(y, tx, initial_w)
    expected_error = y - tx @ expected_w
    expected_loss = (0.5 / len(y)) * (expected_error.T @ expected_error)

    np.testing.assert_allclose(w, expected_w, rtol=1e-7)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-7)


def test_mean_squared_error_sgd(linear_data):
    y, tx, initial_w = linear_data
    np.random.seed(42)
    w, loss = mean_squared_error_sgd(y, tx, initial_w, max_iters=10, gamma=0.01)

    assert w.shape == initial_w.shape
    assert np.isscalar(loss) or loss.ndim == 0
    assert loss >= 0.0


# ---------------------------------------------------------------------------
# 3. Logistic Regression & Regularized Logistic Regression Tests
# ---------------------------------------------------------------------------


def test_sigmoid_docstring():
    res1 = sigmoid(np.array([0.1]))
    res2 = sigmoid(np.array([0.1, 0.1]))

    np.testing.assert_allclose(res1, np.array([0.52497919]), rtol=1e-6)
    np.testing.assert_allclose(res2, np.array([0.52497919, 0.52497919]), rtol=1e-6)
    np.testing.assert_allclose(sigmoid(0.0), 0.5, rtol=1e-7)


def test_calculate_loss_docstring():
    y = np.array([0.0, 1.0])
    tx = np.arange(4).reshape(2, 2)
    w = np.array([2.0, 3.0])

    loss = calculate_loss(y, tx, w)
    np.testing.assert_allclose(loss, 1.52429481, rtol=1e-7)


def test_calculate_gradient_docstring():
    y = np.array([0.0, 1.0])
    tx = np.arange(6).reshape(2, 3)
    w = np.array([0.1, 0.2, 0.3])

    grad = calculate_gradient(y, tx, w)
    expected = np.array([-0.10370763, 0.2067104, 0.51712843])
    np.testing.assert_allclose(grad, expected, rtol=1e-6)


def test_logistic_regression(logistic_data):
    y, tx, initial_w = logistic_data
    gamma = 0.1

    # 0 iterations check
    w0, loss0 = logistic_regression(y, tx, initial_w, max_iters=0, gamma=gamma)
    np.testing.assert_allclose(w0, initial_w)
    np.testing.assert_allclose(loss0, calculate_loss(y, tx, initial_w))

    # 1 iteration check
    w1, loss1 = logistic_regression(y, tx, initial_w, max_iters=1, gamma=gamma)
    expected_w1 = initial_w - gamma * calculate_gradient(y, tx, initial_w)
    expected_loss1 = calculate_loss(y, tx, expected_w1)

    np.testing.assert_allclose(w1, expected_w1, rtol=1e-7)
    np.testing.assert_allclose(loss1, expected_loss1, rtol=1e-7)
    # Loss should decrease after 1 step of GD with small learning rate
    assert loss1 < loss0


def test_reg_logistic_regression(logistic_data):
    y, tx, initial_w = logistic_data
    gamma = 0.1
    lambda_ = 0.5

    # With lambda_ = 0, should behave identically to unregularized logistic_regression
    w_unreg, loss_unreg = logistic_regression(
        y, tx, initial_w, max_iters=5, gamma=gamma
    )
    w_reg0, loss_reg0 = reg_logistic_regression(
        y, tx, 0.0, initial_w, max_iters=5, gamma=gamma
    )
    np.testing.assert_allclose(w_unreg, w_reg0, rtol=1e-7)
    np.testing.assert_allclose(loss_unreg, loss_reg0, rtol=1e-7)

    # 1 iteration check with non-zero lambda_
    w1, loss1 = reg_logistic_regression(
        y, tx, lambda_, initial_w, max_iters=1, gamma=gamma
    )
    expected_grad = calculate_gradient(y, tx, initial_w) + 2 * lambda_ * initial_w
    expected_w1 = initial_w - gamma * expected_grad
    expected_loss1 = calculate_loss(
        y, tx, expected_w1
    )  # Loss is returned without penalty term

    np.testing.assert_allclose(w1, expected_w1, rtol=1e-7)
    np.testing.assert_allclose(loss1, expected_loss1, rtol=1e-7)
