# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
from numpy import dot

from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test,
                                    learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/100
        # set bias

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Write your code to train the perceptron here
        pass
        for epoch in range(self.epochs):
            for i in range(self.trainingSet.input.shape[0]):
                pre_result = dot(self.weight, self.trainingSet.input[i,:])
                result = Activation.sign(pre_result, threshold=0)
                error = self.trainingSet.label[i] - result
                self.updateWeights(self.trainingSet.input[i,:], error)
            correct = 0.0
            for j in range(self.validationSet.input.shape[0]):
                valid_result = Activation.sign(dot(self.weight, self.validationSet.input[j,:]))
                if valid_result == self.validationSet.label[j]:
                    correct += 1.0
            accuracy = correct/self.validationSet.input.shape[0]
           # print('after %d times training, valiation accuracy : %.4f %d' %(epoch, accuracy, correct))
           # Den Schwellwert hab ich selbst auf 0.98 gesetzt
            if accuracy >= 0.98:
                if verbose:
                    print('After %d times training, Validation accuracy:%.4f>0.98' %(epoch, accuracy))
                    print('Stop training to avoid overfitting!')
                break
        if epoch == self.epochs - 1:
            print('No accuracy >= threshold, no need to break loop to avoid overfitting')

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        # Write your code to do the classification on an input image
        pass
        testResult = Activation.sign(dot(self.weight, testInstance), threshold=0)
        return testResult

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, input, error):
        # Write your code to update the weights of the perceptron here
        pass
        self.weight += self.learningRate * error * input

    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return Activation.sign(np.dot(np.array(input), self.weight))
