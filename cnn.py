# -*- coding: utf-8 -*-
"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""

from __future__ import print_function

import os
import sys
import timeit
import cPickle

import numpy
import pandas as pd

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2),
                 W=None, b=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        if W is None:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        self.W = W
        # the bias is a 1D tensor -- one bias per output feature map
        # the number of filters
        if b is None:
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)
        self.b = b
        # convolve input feature maps with filters
        # returns - (batch size, num of filters, output row, output col)
        conv_out = conv2d(
            input=input,
            filters=self.W
        )
        
        # downsample each feature map individually, using maxpooling
        # returns - (batch size, num of filters, pooled row, pooled col)
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


class CNN(object):
    """
    A Two-Convolutional Layer Neural Network.
    
    """
    def __init__(self, rng, input, pixels, filters, poolsize, 
                 nkerns, batch_size, n_in, n_out):     
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape
        
        :type pixels: tuple
        :param pixels: pixels of the image, (height, width, channels)
        
        :type filters: tuple
        :param filters: size of the filters, (height, width)
        
        :type pool: tuple
        :param pool: size of the maxpool, (height, width)
    
        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer. i.e. the number of filters
                       of each convolutional layer
                       
        :type n_in: int
        :param n_in: number of input units into the logistic layer

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        """
    
        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to 
        # (img_height-filter_height+1 , img_height-filter_height+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        self.layer0 = LeNetConvPoolLayer(
            rng,
            input=input,
            image_shape=(batch_size, pixels[2], pixels[0], pixels[1]),
            filter_shape=(nkerns[0], pixels[2], filters[0], filters[1]),
            poolsize=poolsize
        )
        # shape of the output of layer0
        output_shape0 = ((pixels[0]-filters[0]+1) // poolsize[0], 
                        (pixels[1]-filters[1]+1) // poolsize[1])
    
        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        self.layer1 = LeNetConvPoolLayer(
            rng,
            input=self.layer0.output,
            image_shape=(batch_size, nkerns[0], output_shape0[0], output_shape0[1]),
            filter_shape=(nkerns[1], nkerns[0], filters[0], filters[1]),
            poolsize=poolsize
        )
        # shape of the output of layer1
        output_shape1 = ((output_shape0[0]-filters[0]+1) // poolsize[0],
                         (output_shape0[1]-filters[1]+1) // poolsize[1])
    
        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        layer2_input = self.layer1.output.flatten(2)
    
        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * numpy.prod(output_shape1) ,
            n_out=n_in,
            activation=T.tanh
        )
    
        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = LogisticRegression(
            input=self.layer2.output, n_in=n_in, n_out=n_out
        )
        
            # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer3.params + self.layer2.params + \
                      self.layer1.params + self.layer0.params
        self.input = input
        self.pixels = pixels
        
    def negative_log_likelihood(self, y):
        """
        negative log likelihood of the MLP is given by the negative
        log likelihood of the output of the model, computed in the
        logistic regression layer
        """
        return self.layer3.negative_log_likelihood(y)
        
    def errors(self, y):
        """
        same holds for the function computing the number of errors
        """
        return self.layer3.errors(y)
        

def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    file_x='X.csv', file_y='y.csv',
                    pixels=(28, 28, 1), 
                    filters=(5, 5), 
                    poolsize=(2,2), 
                    nkerns=(20, 50),
                    batch_size=200,
                    n_in=500,
                    n_out=10
                    ):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing 
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(file_x, file_y)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, pixels[2], pixels[0], pixels[1]))
    
    classifier = CNN(
                     rng, 
                     layer0_input,
                     pixels, 
                     filters, 
                     poolsize, 
                     nkerns,
                     batch_size,
                     n_in=n_in,
                     n_out=n_out                            
                    )

    # the cost we minimize during training is the NLL of the model
    cost = classifier.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of gradients for all model parameters
    grads = T.grad(cost, classifier.params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(classifier.params, grads)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, cost %.4f, '
                      'validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, cost_ij,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                           
                        # save the best model
                    with open('best_cnn.pkl', 'wb') as f:
                        cPickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break
 
    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

def predict(model, file_x):
    """
    An example of how to load a trained model and use it
    to predict unkown labels.
    
    :params: model - best_model.pkl
    :file_x: input.csv
    """

    # load the saved model
    classifier = cPickle.load(open(model))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.layer3.y_pred)

    # We can test it on some examples from test test
    pred_set_x = pd.read_csv(file_x, header=None).values
    m = pred_set_x.shape[0]
    
    pred_set_x = pred_set_x.reshape(
        (m, 
         classifier.pixels[2], 
         classifier.pixels[0], 
         classifier.pixels[1])
    )

   
    predicted_values = predict_model(pred_set_x)
    print ('Predicted values for the first 10 examples in test set:')
    print (predicted_values[:10])
    pd.DataFrame(predicted_values).to_csv('predict/'+model[:-4]+'_pred.csv', 
                                          header=None, index=None)

if __name__ == '__main__':
    #evaluate_lenet5()
    predict('best_cnn.pkl', 'X_pred.csv')


