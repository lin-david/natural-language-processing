#################################################
# ================ DO NOT MODIFY ================
#################################################
import sys
import math
import numpy as np
import pickle

# convolution window size
width = 2

# number of filters
F = 100

# learning rate
alpha = 1e-1

# vocabsize: size of the total vocabulary
# the text in the input file will be transformed into respective
# positional indices in the vocab dictionary
# as the input for the forward and backward algorithm
# e.g. if vocab = {'hello': 0, 'world': 1} and the training data is
# "hello hello world hello world",
# the input to the forward and backward algorithm will be [0, 0, 1, 0, 1]
vocabsize = 10000
vocab = {}

np.random.seed(1)

# U and V are weight vectors of the hidden layer
# U: a matrix of weights of all inputs for the first
# hidden layer for all F filters in the
# where each filter has the size of vocabsize by width
# U[i, j, k] represents the weight of filter u_j
# for word with vocab[word] = i when the word is
# in the position offset k of the sliding window
# e.g. in our earlier example of "hello hello world hello world",
# if the window size is 4 and we are looking at the first sliding window
# of the 9th filter, the weight for the last "hello" will be U[0, 8, 3]
U = np.random.normal(loc=0, scale=0.01, size=(vocabsize, F, width))

# V: the the weight vector of the F filter outputs (after max pooling)
# that will produce the output, i.e. o = sigmoid(V*h)
V = np.random.normal(loc=0, scale=0.01, size=(F))


def sigmoid(x):
    """
    helper function that computes the sigmoid function
    """
    return 1. / (1 + math.exp(-x))


def read_vocab(filename):
    """
    helper function that builds up the vocab dictionary for input transformation
    """
    file = open(filename)
    for line in file:
        cols = line.rstrip().split("\t")
        word = cols[0]
        idd = int(cols[1])
        vocab[word] = idd
    file.close()


def read_data(filename):
    """
    :param filename: the name of the file
    :return: list of tuple ([word index list], label)
    as input for the forward and backward function
    """
    data = []
    file = open(filename)
    for line in file:
        cols = line.rstrip().split("\t")
        label = int(cols[0])
        words = cols[1].split(" ")
        w_int = []
        for w in words:
            # skip the unknown words
            if w in vocab:
                w_int.append(vocab[w])
        data.append((w_int, label))
    file.close()
    return data


def train():
    """
    main caller function that reads in the names of the files
    and train the CNN to classify movie reviews
    """
    vocabFile = sys.argv[2]
    trainingFile = sys.argv[3]
    testFile = sys.argv[4]

    read_vocab(vocabFile)
    training_data = read_data(trainingFile)
    test_data = read_data(testFile)

    for i in range(50):
        # confusion matrix showing the accuracy of the algorithm
        confusion_training = np.zeros((2, 2))
        confusion_validation = np.zeros((2, 2))

        for (data, label) in training_data:
            # back propagation to update weights for both U and V
            backward(data, label)

            # calculate forward and evaluate
            prob = forward(data)["prob"]
            pred = 1 if prob > .5 else 0
            confusion_training[pred, label] += 1

        for (data, label) in test_data:
            # calculate forward and evaluate
            prob = forward(data)["prob"]
            pred = 1 if prob > .5 else 0
            confusion_validation[pred, label] += 1

        print("Epoch: {}\tTrain accuracy: {:.3f}\tDev accuracy: {:.3f}"
            .format(
            i,
            np.sum(np.diag(confusion_training)) / np.sum(confusion_training),
            np.sum(np.diag(confusion_validation)) / np.sum(confusion_validation)))


#################################################
# ========= IMPLEMENT FUNCTIONS BELOW ===========
#################################################

def forward(word_indices):
    """
    :param word_indices: a list of word indices, i.e. idx = vocab[word]
    :return: a result dictionary containing 3 fields -

    result['prob']:
    output of the CNN algorithm. predicted probability of 1

    result['h']:
    the hidden layer output after max pooling, h = [h1, ..., hF]

    result['hid']:
    argmax of F filters, e.g. j of x_j
    e.g. for the ith filter u_i, tanh(word[hid[i], hid[i] + width]*u_i) = max(h_i)
    """

    h = np.zeros(F, dtype=float)
    hid = np.zeros(F, dtype=int)
    prob = 0

    # step 1. compute h and hid
    # loop through the input data of word indices and
    # keep track of the max filtered value h_i and its position index x_j
    # h_i = max(tanh(weighted sum of all words in a given window)) over all windows for u_i
    """
    Type your code below
    """
    for j in range(0, F):
        p_per_filter = np.zeros(len(word_indices) - width + 1, dtype=float)
        maxi = float("-inf")
        argmaxi = -1
        for i in range(0, len(word_indices) - width + 1):
            vec_u_sum = U[word_indices[i + 0], j, 0] + U[word_indices[i + 1], j, 1]

            result = np.tanh(vec_u_sum)

            if result > maxi:
                maxi = result
                argmaxi = i

        h[j] = maxi
        hid[j] = argmaxi

    # step 2. compute probability
    # once h and hid are computed, compute the probabiliy by sigmoid(h^TV)
    """
    Type your code below
    """
    prob = sigmoid(np.dot(np.transpose(h), V))

    # step 3. return result
    return {"prob": prob, "h": h, "hid": hid}

def calc_ana_V_grad(word_indices, true_label, prob, h):
    V_grad_update = np.zeros(F, dtype=float)

    for j in range(0, len(h)):
        V_grad_update[j] = (true_label - prob) * h[j]

    return V_grad_update

def calc_ana_U_grad(word_indices, true_label, prob, h, hid):
    U_grad_update = np.zeros((F, width), dtype=float)

    sech2 = lambda x: 1/np.square(np.cosh(x))

    for j in range(0, len(hid)):
        i = hid[j]

        vec_u_sum = U[word_indices[i + 0], j, 0] + U[word_indices[i + 1], j, 1]

        for k in range(0, width):
            dh_du = sech2(vec_u_sum)
            U_grad_update[j, k] = (true_label - prob) * V[j] * dh_du

    return U_grad_update

def backward(word_indices, true_label):
    """
    :param word_indices: a list of word indices, i.e. idx = vocab[word]
    :param true_label: true label (0, 1) of the movie reviews
    :return: None

    update weight matrix/vector U and V based on the loss function
    """
    global U, V
    pred = forward(word_indices)
    prob = pred["prob"]
    h = pred["h"]
    hid = pred["hid"]

    # update U and V here
    # loss_function = y * log(o) + (1 - y) * log(1 - o)
    #               = true_label * log(prob) + (1 - true_label) * log(1 - prob)
    # to update V: V_new = V_current + d(loss_function)/d(V)*alpha
    # to update U: U_new = U_current + d(loss_function)/d(U)*alpha
    # Make sure you only update the appropriate argmax term for U
    """
    Type your code below
    """
    for j in range(0, len(h)):
        V_grad_update = (true_label - prob) * h[j]
        V[j] = V[j] + V_grad_update * alpha

    sech2 = lambda x: 1/np.square(np.cosh(x))

    for j in range(0, len(hid)):
        i = hid[j]

        vec_u_sum = U[word_indices[i + 0], j, 0] + U[word_indices[i + 1], j, 1]

        dh_du = sech2(vec_u_sum)
        U_grad_update = (true_label - prob) * V[j] * dh_du
        U[word_indices[i + 0], j, 0] = U[word_indices[i + 0], j, 0] + U_grad_update * alpha
        U[word_indices[i + 1], j, 1] = U[word_indices[i + 1], j, 1] + U_grad_update * alpha


def calc_numerical_gradients_V(word_indices, true_label):
    """
    :param true_label: true label of the data
    :param V: weight vector of V
    :param word_indices: a list of word indices, i.e. idx = vocab[word]
    :return V_grad:
    V_grad =    a vector of size length(V) where V_grad[i] is the numerical
                gradient approximation of V[i]
    """
    # you might find the following variables useful
    x = word_indices
    y = true_label
    eps = 1e-4
    V_grad = np.zeros(F, dtype=float)

    """
    Type your code below
    """
    global U, V
    V_og = np.copy(V)

    for j in range(0, len(V)):
        V[j] += eps
        pred = forward(x)
        prob = pred["prob"]
        J_plus = y * np.log(prob) + (1 - y) * np.log(1 - prob)

        V[j] = V_og[j]
        V[j] -= eps
        pred = forward(x)
        prob = pred["prob"]
        J_minus = y * np.log(prob) + (1 - y) * np.log(1 - prob)

        V[j] = V_og[j]
        V_grad[j] = (J_plus - J_minus) / (2 * eps)

    return V_grad


def calc_numerical_gradients_U(word_indices, true_label):
    """
    :param U: weight matrix of U
    :param word_indices: a list of word indices, i.e. idx = vocab[word]
    :param true_label: true label of the data
    :return U_grad:
    U_grad =    a matrix of dimension F*width where U_grad[i, j] is the numerical
                approximation of the gradient for the argmax of
                each filter i at offset position j
    """
    # you might find the following variables useful
    x = word_indices
    y = true_label
    eps = 1e-4

    pred = forward(x)
    prob = pred["prob"]
    h = pred["h"]
    hid = pred["hid"]
    U_grad = np.zeros((F, width))

    """
    Type your code below
    """
    global U, V
    U_og = np.copy(U)

    for j in range(0, len(hid)):
        i = hid[j]

        for k in range(0, width):
            U[word_indices[i + k], j, k] += eps
            pred = forward(x)
            prob = pred["prob"]
            J_plus = y * np.log(prob) + (1 - y) * np.log(1 - prob)

            U[word_indices[i + k], j, k] = U_og[word_indices[i + k], j, k]
            U[word_indices[i + k], j, k] -= eps
            pred = forward(x)
            prob = pred["prob"]
            J_minus = y * np.log(prob) + (1 - y) * np.log(1 - prob)

            U[word_indices[i + k], j, k] = U_og[word_indices[i + k], j, k]
            U_grad[j, k] = (J_plus - J_minus) / (2 * eps)

    return U_grad


def check_gradient():
    """
    :return (diff in V, diff in U)
    Calculate numerical gradient approximations for U, V and
    compare them with the analytical values
    check gradient accuracy; for more details, cf.
    http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
    """
    x = []
    for i in range(100):
        x.append(np.random.randint(vocabsize))
    y = 1

    pred = forward(x)
    prob = pred["prob"]
    h = pred["h"]
    hid = pred["hid"]

    """
    Update 0s below with your calculations
    """
    # check V
    # compute analytical and numerical gradients and compare their differences
    ana_grad_V = calc_ana_V_grad(x, y, prob, h) # <-- Update
    numerical_grad_V = calc_numerical_gradients_V(x, y) # <-- Update
    sum_V_diff = sum((numerical_grad_V - ana_grad_V) ** 2)

    # check U
    # compute analytical and numerical gradients and compare their differences
    ana_grad_U = calc_ana_U_grad(x, y, prob, h, hid) # <-- Update
    numerical_grad_U = calc_numerical_gradients_U(x, y) # <-- Update
    sum_U_diff = sum(sum((numerical_grad_U - ana_grad_U) ** 2))

    print("V diff: {:.24f}, U diff: {:.8f} (these should be close to 0)"
          .format(sum_V_diff, sum_U_diff))



#################################################
# ================ DO NOT MODIFY ================
#################################################

def load_gradient_vars():
    with open('grad.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)

    return data.U, data.word_indices, data.true_label


def load_gradient_vars():
    with open('grad.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)

    return data.U, data.word_indices, data.true_label


def load_forward_vars():
    with open('grad.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)

    return data.word_indices

def load_backward_vars():
    with open('grad.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)

    return data.word_indices, data.true_label

# run the entire file with:
# python hw2_cnn.py -t vocab.txt movie_reviews.train movie_reviews.dev
#
if __name__ == "__main__":
    if sys.argv[1] == "-t":
        check_gradient()
        train()
    elif sys.argv[1] == "-g":
        U, word_indices, true_label = load_gradient_vars()
        calc_numerical_gradients_U(word_indices, true_label)
    elif sys.argv[1] == "-f":
        word_indices = load_forward_vars()
        forward(word_indices)
    elif sys.argv[1] == "-b":
        word_indices, true_label = load_backward_vars()
        backward(word_indices, true_label)
    else:
        print("Usage: python hw2_cnn.py -t vocab.txt movie_reviews.train movie_reviews.dev")
