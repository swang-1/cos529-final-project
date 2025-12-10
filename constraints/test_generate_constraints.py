import z3
import numpy as np
from generate_constraints import *


def generate_weights(pixels_per_image, hidden_size, num_labels, n_hidden):
    rng = np.random.default_rng()

    weights, biases = [], []

    weights.append(0.2 * rng.random((pixels_per_image, hidden_size)) - 0.1)
    biases.append(0.2 * rng.random(hidden_size) - 0.1)

    for _ in range(n_hidden - 1):
        weights.append(0.2 * rng.random((hidden_size, hidden_size)) - 0.1)
        biases.append(0.2 * rng.random(hidden_size) - 0.1)

    weights.append(0.2 * rng.random((hidden_size, num_labels)) - 0.1)
    biases.append(0.2 * rng.random(num_labels))

    return weights, biases

def test_generate_constraints_individually():
    weights = [
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[2, 2, 2], [1, 1, 1], [2, 2, 2]])
    ]

    biases = [
        np.array([1, 2, 3]),
        np.array([1, 2, 3])
    ]

    input_size = len(weights[0])
    layer_sizes = [len(weights[i][0]) for i in range(len(weights))]

    xs = get_xs(input_size, layer_sizes[:-1])
    ys = get_ys(layer_sizes)

    print(f"x variables: \n{xs}")
    print(f"y variables: \n{ys}\n")

    input_const = generate_input_constraints(xs)
    print("Generate input constraints:")
    print(f"{input_const}\n")

    relu_const = generate_relu_constraints(xs, ys)
    print("Generate relu constraints:")
    print(f"{relu_const}\n")

    linear_const = generate_linear_layer_constraints(xs, ys, weights, biases)
    print("Generate linear layer constraints:")
    print(f"{linear_const}")

def test_generate_constraints():
    weights, biases = generate_weights(100, 10, 10, 4)

    res, _, _ = generate_constraints(weights, biases)
    print(res)

if __name__ == "__main__":
    # test_generate_constraints_individually()
    test_generate_constraints()