import z3
import numpy as np
from tqdm import tqdm


def generate_constraints(weights, biases):
    '''
    Generate all constraints for the FFNN describes by the given
    weights and biases.
    
    Inputs:
        weights: A list of matrices where each matrix contains the
            linear weights of each layer
        biases: A list of vectors where each vector represents the
            linear bias of each layer
    Output:
        res: A list of constraints for the FFNN described by weights
            and biases that can be added to a solver
    '''
    input_size = len(weights[0])
    layer_sizes = [len(weights[i][0]) for i in range(len(weights))]

    xs = get_xs(input_size, layer_sizes[:-1])
    ys = get_ys(layer_sizes)

    res = []

    res += generate_input_constraints(xs)
    res += generate_linear_layer_constraints(xs, ys, weights, biases)
    res += generate_relu_constraints(xs, ys)

    return res, xs, ys


def generate_constraints_no_nonlinearity(weights, biases):
    '''
    Generate all constraints for the FFNN describes by the given
    weights and biases with no non-linearity.
    
    Inputs:
        weights: A list of matrices where each matrix contains the
            linear weights of each layer
        biases: A list of vectors where each vector represents the
            linear bias of each layer
    Output:
        res: A list of constraints for the FFNN described by weights
            and biases that can be added to a solver
    '''
    input_size = len(weights[0])
    layer_sizes = [len(weights[i][0]) for i in range(len(weights))]

    xs = get_xs(input_size, layer_sizes[:-1])
    ys = get_ys(layer_sizes)

    res = []

    res += generate_input_constraints(xs)
    res += generate_linear_layer_constraints(xs, ys, weights, biases)
    res += generate_id_activation_constraints(xs, ys)

    return res, xs, ys


def get_xs(input_size, layer_sizes):
    '''
    Returns the matrix of Z3 variables for an n-layer FFNN.
    xs[0] represents the input vector and xs[i] for 1 <= i <= n
    represents the vector after applying the transformation induced
    by layer i and the relevant activation function.
    
    Inputs:
        input_size: The size of the input vector
        layer_sizes: A list of Int values where layer_sizes[i] is the
            length of the vector output at layer i
    Outputs:
        xs: An array of Z3 variables where xs[i] represents the vector
            after applying the transformation/activation function
            at layer i
    '''
    xs = []
    
    x0 = [z3.Real(f'x_0_{i}') for i in range(input_size)]
    xs.append(x0)

    for i in range(len(layer_sizes)):
        xi = [z3.Real(f'x_{i + 1}_{j}') for j in range(layer_sizes[i])]
        xs.append(xi)

    return xs

def get_ys(layer_sizes):
    '''
    Returns a matrix of Z3 variables for intermediate values of an n-layer FFNN.
    Specifically, ys[i] represents the intermediate vector after applying the
    linear transformation at layer i and before applying the activation function.
    
    Input:
        layer_sizes: A list of Int values where layer_sizes[i] is the
            length of the vector output at layer i
    Output:
        ys: An array of Z3 variables where ys[i] represents the vector
            after applying the linear transformation at layer i
    '''
    ys = [None]

    for i in range(len(layer_sizes)):
        yi = [z3.Real(f'y_{i + 1}_{j}') for j in range(layer_sizes[i])]
        ys.append(yi)

    return ys


def generate_input_constraints(xs, lower=0, upper=1):
    '''
    Generate the constraints on the input of the neural network.
    The constraint is that each entry of the input vector is within
    some bounds. In other words, given lower and upper bounds l and u, 

        l <= x_0_i <= u

    for all 0 <= i < input_len

    Inputs:
        xs: the array of Z3 variables representing the input vector
        lower: the lower bound for entries of the input vector
        upper: the upper bound for entries of the input vector
    Output:
        res: The list of Z3 solver constraints corresponding to the input
    '''
    res = []
    for x in tqdm(xs[0]):
        res.append(x >= lower)
        res.append(x <= upper)

    return res

def generate_relu_constraints(xs, ys):
    '''
    Generate the constraints corresponding to the relu 
    application to the output at each layer. That is, for layer i with
    output size n,

        x_i_j = max(0, y_i_j)

    where y_i_j denotes the output after layer i and 0 <= j <= len(layer_i)

    Input:
        xs: An array of Z3 variables where xs[i] represents the vector
            after applying the transformation/activation function
            at layer i
        ys: An array of Z3 variables where ys[i] represents the vector
            after applying the linear transformation at layer i
    Output:
        res: TThe list of Z3 solver constraints corresponding to the relu applications
    '''
    res = []
    for i in tqdm(range(1, len(ys) - 1)):
        for j in range(len(ys[i])):
            constraint = xs[i][j] == z3.If(ys[i][j] >= 0, ys[i][j], 0)
            res.append(constraint)

    return res

def generate_id_activation_constraints(xs, ys):
    '''
    Generate the constraints corresponding to the identity activation function.
    These constraints are for experiments 

    Input:
        xs: An array of Z3 variables where xs[i] represents the vector
            after applying the transformation/activation function
            at layer i
        ys: An array of Z3 variables where ys[i] represents the vector
            after applying the linear transformation at layer i
    Output:
        res: TThe list of Z3 solver constraints corresponding to the relu applications
    '''
    res = []
    for i in tqdm(range(1, len(ys) - 1)):
        for j in range(len(ys[i])):
            constraint = xs[i][j] == ys[i][j]
            res.append(constraint)

    return res

def generate_linear_layer_constraints(xs, ys, W, B):
    '''
    Generate the constraints corresponding to the application of each linear
    layer in the NN. Specifically, for 0 <= i < n:

        ys[i + 1] = W[i]xs[i] + B[i]

    Where W and B are the weights and biases at each layer respectively.

    Input:
        xs: An array of Z3 variables where xs[i] represents the vector
            after applying the transformation/activation function
            at layer i
        ys: An array of Z3 variables where ys[i] represents the vector
            after applying the linear transformation at layer i
        W: A matrix where W[i] are the linear weights at layer i
        B: A matrix where B[i] are the linear biase terms at layer i 
    '''
    res = []

    for i in tqdm(range(len(W))):
        w = W[i]
        b = B[i]
        x = xs[i]
        y = ys[i + 1]

        for j in range(len(y)):
            val = var_dot_product(w[:,j], x) + b[j]
            res.append(y[j] == val )
        
    return res


def var_dot_product(v1, v2):
    '''
    Returns the dot product of v1 and v2 as a Z3 term
    
    Inputs:
        v1: The first vector of numbers/z3 terms to take the dot product with
        v2: The second vector of numbers/z3 terms to take the dot product with
    Output:
        dot: A z3 term representing the dot product of v1 and v2
    '''
    assert len(v1) > 0, "vector v1 is empty"
    assert len(v2) > 0, "vector v2 is empty"
    assert len(v1) == len(v2), "vectors v1 and v2 have different lengths"

    dot = v1[0] * v2[0]
    for i in range(1, len(v1)):
        dot += (v1[i] * v2[i])

    return dot