from linear_genetic_programming._evolve import Evolve
from linear_genetic_programming._population import Population
import numpy as np
import copy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class LGPClassifier(BaseEstimator, ClassifierMixin):
    '''
    A Linear Genetic Programming classifier

    Parameters
    ----------
    numberOfInput : integer, required
        number of features, X.shape[1]

    numberOfOperation: integer, optional
        +, -, *, /, ^, if less, if more

    numberOfVariable: integer, optional (default=4)
        A variable number of additional registers used to aid in calculations performed
        as part of a program. Generally, these are initialized with a default value
        before a program is executed.



    tournament_size : integer, optional (default=4)
        The number of programs that will compete to become part of the next
        generation.

    maxGenerations : integer, optional (default=1000)
        The number of generations to evolve.

    evolutionStrategy: "population" or "steady state"
        population: traditional genetic algorithm

    '''

    def __init__(self,
                 numberOfInput = 3,
                 numberOfOperation = 5,
                 numberOfVariable = 4,
                 numberOfConstant = 9,
                 macro_mutate_rate = 0.5,
                 max_prog_ini_length = 30,
                 min_prog_ini_length = 10,
                 maxProgLength = 200,
                 minProgLength = 10,
                 pCrossover = 0.75,
                 pConst = 0.5,
                 pInsert = 0.5,
                 pRegmut = 0.1,
                 pMacro = 0.75,
                 pMicro = 0.5,
                 tournamentSize = 2,
                 maxGeneration = 2,
                 fitnessThreshold = 1.0,
                 populationSize = 1000,
                 showGenerationStat = True,
                 randomSampling = True,
                 evolutionStrategy = "steady state"):
        self.numberOfInput = numberOfInput
        self.numberOfOperation = numberOfOperation
        self.numberOfVariable = numberOfVariable
        self.numberOfConstant = numberOfConstant
        self.macro_mutate_rate = macro_mutate_rate
        self.max_prog_ini_length = max_prog_ini_length
        self.min_prog_ini_length = min_prog_ini_length
        self.maxProgLength = maxProgLength
        self.minProgLength = minProgLength
        self.pCrossover = pCrossover
        self.pConst = pConst
        self.pInsert = pInsert
        self.pRegmut = pRegmut
        self.pMacro = pMacro
        self.pMicro = pMicro
        self.tournamentSize = tournamentSize
        self.maxGeneration = maxGeneration
        self.fitnessThreshold = fitnessThreshold
        self.populationSize = populationSize
        self.showGenerationStat = showGenerationStat
        self.randomSampling = randomSampling
        self.evolutionStrategy = evolutionStrategy

    #   register numberOfInput + numberOfVariable + numberOfConstant
    def __generateRegister(self):
        register_length = self.numberOfInput + self.numberOfVariable + self.numberOfConstant
        register = np.zeros(register_length, dtype=float)
        for i in range(self.numberOfVariable + self.numberOfInput):
            register[i] = 2 * np.random.random_sample() - 1 # random float [-1, 1)
        # initialize constant
        j = self.numberOfVariable + self.numberOfInput
        while j < register_length:
            register[j] = np.around(np.random.choice(np.arange(-1, 1.1, 0.1)), 2) #j - self.numberOfVariable + self.numberOfInput + 1
            j += 1
        self.register_ = register

    def fit(self, X, y):
        """
        Fit the Genetic Program according to X, y.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        classes_ : ndarray, shape (n_classes,)
            The classes seen at :meth:`fit`.
        Returns
        -------
        self : best program for classification
            Returns self.
        """

        X, y = check_X_y(X, y)
        self.is_fitted_ = True
        self.classes_ = unique_labels(y)
        if len(self.classes_) == 1:
            raise ValueError("y is unbalanced, only have one class")

        self.__generateRegister()
        # generate random population
        p = Population()
        p.generatePopulation(self.numberOfOperation, self.numberOfVariable, self.numberOfInput, self.numberOfConstant,
                             self.pConst, self.max_prog_ini_length, self.min_prog_ini_length, self.populationSize)

        e = Evolve(self.tournamentSize, self.maxGeneration, p)  # tournamentSize, maxGeneration, population
        bestProg = e.evolveGeneration(self.pRegmut, self.pMicro, self.pMacro, self.pConst, self.pCrossover,
                                      self.numberOfVariable, self.numberOfInput, self.numberOfOperation,
                                      self.numberOfConstant, self.register_, self.pInsert, self.maxProgLength,
                                      self.minProgLength, X, y, self.fitnessThreshold, self.showGenerationStat,
                                      self.randomSampling, self.evolutionStrategy)
        self.bestProgStr_ = bestProg.toString(self.numberOfVariable, self.numberOfInput, self.register_)
        effProg = copy.deepcopy(bestProg)
        effProg = effProg.eliminateStrcIntron()
        self.bestEffProgStr_ = effProg.toString(self.numberOfVariable, self.numberOfInput, self.register_)
        self.bestProg_ = bestProg
        self.bestProFitness_ = round(bestProg.fitness, 4)
        self.populationAvg_ = round(e.p.getAverageFitness(), 4)

        # `fit` should always return `self`
        return self

    def predict(self, X):
        '''
        Use best program to predict y
        :param X_test:
        :return:
        '''
        X = check_array(X)
        check_is_fitted(self)
        classType = self.classes_[0].dtype
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=classType)
        for i, row in enumerate(X):
            pred = self.bestProg_.predictProbaSigmoid(self.numberOfVariable, self.register_, row)
            if pred <= 0.5:  # class A
                y_pred[i] = self.classes_[0]
            else:  # class B
                y_pred[i] = self.classes_[1]
        return y_pred

    # may wrong in return format, check it later
    def predict_proba(self, X):
        """
        Probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           Vector to be scored, where `n_samples` is the number of samples and
           `n_features` is the number of features.
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
           Returns the probability of the sample for each class in the model,
           where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self)

        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=np.float64)
        for i, row in enumerate(X):
            y_pred[i] = self.bestProg_.predictProbaSigmoid(self.numberOfVariable, self.register_, row)
        return y_pred

    def _more_tags(self):
        return {'binary_only': True}

