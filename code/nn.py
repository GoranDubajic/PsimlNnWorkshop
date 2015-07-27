import numpy

from logistic_regression import LogisticRegression, load_data

class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, W=None, b=None,
                 activation=numpy.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_out, n_in)
                ),
                dtype=numpy.float32
            )

        if b is None:
            b_values = numpy.zeros((n_out, 1), dtype=numpy.float32)

        self.W = W_values
        self.b = b_values
        self.activation = activation

    def forward(self, x):
        # Calculate forward for pass for this layer.
        lin_output = numpy.dot(self.W, x) + self.b
        # If activation is not None apply it on pre-activations
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input = x

    def W_grad(self):
        # Calculate loss function gradient with respect to weight using gradient with respect to
        # pre-activations and input to this layer.
        return numpy.dot(self.pre_activation_grad, numpy.transpose(self.input))

    def b_grad(self):
        # Return loss function gradient with respect to bias.
        return self.pre_activation_grad

class NN(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function BUT BACKWARD NEEDS TO BE UPDATED.
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            n_in=n_in,
            n_out=n_hidden,
            activation=numpy.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer.
        self.logRegressionLayer = LogisticRegression(
            n_in=n_hidden,
            n_out=n_out
        )

    def forward(self, x):
        # Forward pass through the network: forward for logistic regression and hidden layer.
        self.hiddenLayer.forward(x)
        self.logRegressionLayer.forward(self.hiddenLayer.output)

    def backward(self, y):
        # Backward for logistic regression.
        self.logRegressionLayer.calc_pre_activation_grad(y)
        # Calculate loss function hidden layer output gradient.
        self.hiddenLayer.output_grad = numpy.dot(
            numpy.transpose(self.logRegressionLayer.W),
            self.logRegressionLayer.pre_activation_grad)
        # Calculate activation function derivative.
        activation_der = numpy.ones(self.hiddenLayer.output.shape, dtype=numpy.float32) - numpy.square(self.hiddenLayer.output)
        self.hiddenLayer.pre_activation_grad = numpy.multiply(self.hiddenLayer.output_grad, activation_der)

    def update_weights(self, alpha):
        # Update weight for logistic regression and hidden layer.
        self.logRegressionLayer.W -= alpha * self.logRegressionLayer.W_grad()
        self.logRegressionLayer.b -= alpha * self.logRegressionLayer.b_grad()
        self.hiddenLayer.W -= alpha * self.hiddenLayer.W_grad()
        self.hiddenLayer.b -= alpha * self.hiddenLayer.b_grad()

    def test(self, dataset):
        # Input is data set.
        # Classify every example from data set and return two values
        # - Number of misclassified examples over all examples.
        # - List of triplets which are [example index, correct (i.e. expected) class, predicted class]
        x, y = dataset
        errors =[]
        for i in xrange(x.shape[0]):
            self.forward(x[i].reshape(x.shape[1], 1))
            if (self.logRegressionLayer.error(y[i])):
                errors.append([i, y[i], self.logRegressionLayer.y_pred])
        return float(len(errors)) / x.shape[0], errors

def nn_stochastic_gradient_descent(dataset=r'..\data\mnist.pkl.gz', n_epochs=100, alpha=0.01):
        train_set, valid_set, test_set = load_data(dataset)
        # Initialize neural network.
        nn = NN(numpy.random, 28 * 28, 100, 10)
        # Print header.
        print('Epoch\tTrainigError%%\tValidationError%%\tTestError%%')
        # Train network for limited number of epochs.
        for epoch in xrange(n_epochs):
            x, y = train_set
            for i in xrange(x.shape[0]):
                input = x[i].reshape(x.shape[1], 1)
                nn.forward(input)
                nn.backward(y[i])
                nn.update_weights(alpha)
            # Measure accuracy on all data sets.
            train_error, train_errors = nn.test(train_set)
            valid_error, valid_errors = nn.test(valid_set)
            test_error, test_errors = nn.test(test_set)
            print ('%d\t%f\t%f\t%f' %(epoch, 100 * train_error, 100 * valid_error, 100 * test_error))


if __name__ == "__main__":
    nn_stochastic_gradient_descent()