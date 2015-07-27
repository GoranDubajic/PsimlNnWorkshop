import numpy
import numpy.linalg

import gzip
import cPickle

def load_data(dataset):
    # Load the dataset which contains three subsets: training, validation and test (i.e. blind)
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

class LogisticRegression(object):
    """Multi-class Logistic Regression Class
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, n_in, n_out):
                # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_out, n_in)
        self.W = numpy.zeros((n_out, n_in), dtype=numpy.float32)
        # initialize the baises b as a vector of n_out 0s
        self.b = numpy.zeros((n_out, 1), dtype=numpy.float32)

    def forward(self, x):
        # Calculate pre-activation.
        self.pre_activation = 
        # Calculate softmax.
        exp = 
        assert not numpy.isinf(exp).any()
        norm = 
        self.p_y_given_x = 
        # Which class is the most probable?
        self.y_pred = 
        # end-snippet-1
        self.input = x

    def negative_log_likelihood(self, y):
        """Negative log likelihood is the function we optimize i.e. search for a minimum."""
        return 

    def error(self, y):
        # Return true if correctly predicted and false otherwise.
        if self.y_pred == y:
            return 
        else:
            return 

    def calc_pre_activation_grad(self, expected):
        # Calculate loss function gradient with respect to pre-activations.
        self.
        self.

    def W_grad(self):
        # Calculate loss function gradient with respect to weights using pre-activation gradient and input.
        return 

    def b_grad(self):
        # Return loss function gradient with respect to bias.
        return 

    def test(self, dataset):
        # Input is data set.
        # Classify every example from data set and return two values
        # - Number of misclassified examples over all examples.
        # - List of triplets which are [example index, correct (i.e. expected) class, predicted class]
        x, y = dataset
        errors =[]
        Missing stuff!
        return float(len(errors)) / x.shape[0], errors

def stochastic_gradient_descent(dataset=r'..\data\mnist.pkl.gz', n_epochs=100, alpha=0.01):
        train_set, valid_set, test_set = load_data(dataset)
        log_reg = LogisticRegression(28 * 28, 10)
        # Print header.
        print('Epoch\tTrainigError%%\tValidationError%%\tTestError%%')
        # Train for limited number of epochs
        for epoch in xrange(n_epochs):
            x, y = train_set
            for i in xrange(x.shape[0]):
                # First reshape from row to column vector.
                input = x[i].reshape(x.shape[1], 1)
                # Forward pass i.e. prediction.
                log_reg.forward(input)
                # Calculate loss function derivative with respect to pre-activations
                log_reg.calc_pre_activation_grad(y[i])
                # Calculate loss function gradients with respect to weights and update weights.
                log_reg.W -= alpha * log_reg.W_grad()
                log_reg.b -= alpha * log_reg.b_grad()
            # Test accuracy on all data sets.
            train_error, train_errors = log_reg.test(train_set)
            valid_error, valid_errors = log_reg.test(valid_set)
            test_error, test_errors = log_reg.test(test_set)
            print ('%d\t%f\t%f\t%f' %(epoch, 100 * train_error, 100 * valid_error, 100 * test_error))


if __name__ == "__main__":
    stochastic_gradient_descent()
