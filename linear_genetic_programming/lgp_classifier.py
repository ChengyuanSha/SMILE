from linear_genetic_programming._evolve import Evolve
from linear_genetic_programming._population import Population
import numpy as np
import copy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class LGPClassifier(BaseEstimator, ClassifierMixin):
    '''
    Linear Genetic Programming algorithm with scikit learn inspired API.

    Parameters
    ----------
    numberOfInput : integer, required
        Number of features, can be obtained use X.shape[1]

    numberOfOperation: integer, optional
        Operation consists of (+, -, *, /, ^) and branch (if less, if more)

    numberOfVariable: integer, optional (default=4)
        A variable number of additional registers used to aid in calculations performed
        as part of a program. Generally, these are initialized with a default value
        before a program is executed.

    numberOfConstant: int, optional
        Number of constant in register. Constants are stored in registers that are
        write-protected. Constant registers are only initialized once at the beginning
        with values from a constInitRange.

    max_prog_ini_length: int, optional
        Max program initialization length

    min_prog_ini_length: int, optional
        Min program initialization length

    maxProgLength: int, optional
        maximum program length limit during evolution.

    minProgLength: int, optional
        minimum program length required during evolution.

    pCrossover: float, optional
        Probability of exchanging the genetic information of two parent programs

    pConst: float, optional
        Control the probability of constant in Instruciton initialization. It controls whether the
        register will be a constant. It also controls mutation probability in micromutaion. It controls
        whether a register will mutate to constant.

    pInsert: float, optional
        Control probability of insertion in macromutation. It will insert a random instruction
        into the program.

    pRegmut: float, optional
        Control probability of register mutation used in micromutaion. It will either mutate
        register1, register2 or return register.

    pMacro: float, optional
        Probability of macromutation, Macromutation operate on the level of program. It will add
        or delete instruction. It will affect program size.

    pMicro: float, optional
        Probability of micromuation. Micromuation operate on the level of instruction components
        (micro level) and manipulate registers, operators, and constants.

    tournament_size : integer, optional
        The size of tournament selection. The number of programs that will compete to become part of the next
        generation.

    maxGenerations : integer, optional
        The number of generations to evolve.

    fitnessThreshold: float, optional
        When not using random sampling, terminate the evolution if threshold is met'

    populationSize: int, optional
        Size of population

    showGenerationStat: boolean, optional
        If True, print out statistic in each generation

    isRandomSampling: Boolean, optional
        Train the genetic algorithm on random sampled dataset (without replacement)

    evolutionStrategy: "population" or "steady state", optional
        Population: traditional genetic algorithm

    constInitRange: tuple (start, stop, step), optional
        Initiation of the constant set. [start, stop)

    Attributes
    ----------
    register_: array of shape (numberOfInput + numberOfVariable + numberOfConstant, )
        Register stores the calculation variables, feature values and constants.

    bestProg_: class Program
        A list of Instructions used for classification calculation

    bestProFitness_ : float
        Training set accuracy score of the best program

    bestProgStr_: str
        String representation of the best program

    bestEffProgStr_: str
        Intron removed program

    populationAvg_: float
        Average fitness of the final generation
    '''

    def __init__(self,
                 numberOfInput,
                 numberOfOperation = 5,
                 numberOfVariable = 4,
                 numberOfConstant = 9,
                 max_prog_ini_length = 30,
                 min_prog_ini_length = 10,
                 maxProgLength = 200,
                 minProgLength = 10,
                 pCrossover = 0.75,
                 pConst = 0.5,
                 pInsert = 0.5,
                 pRegmut = 0.6,
                 pMacro = 0.75,
                 pMicro = 0.5,
                 tournamentSize = 2,
                 maxGeneration = 200,
                 fitnessThreshold = 1.0,
                 populationSize = 1000,
                 showGenerationStat = True,
                 isRandomSampling = True,
                 evolutionStrategy = "steady state",
                 constInitRange = (1, 11, 1) ):
        self.numberOfInput = numberOfInput
        self.numberOfOperation = numberOfOperation
        self.numberOfVariable = numberOfVariable
        self.numberOfConstant = numberOfConstant
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
        self.isRandomSampling = isRandomSampling
        self.evolutionStrategy = evolutionStrategy
        self.constInitRange = constInitRange

    def __generateRegister(self):
        #   register numberOfInput + numberOfVariable + numberOfConstant
        register_length = self.numberOfInput + self.numberOfVariable + self.numberOfConstant
        register = np.zeros(register_length, dtype=float)
        for i in range(self.numberOfVariable + self.numberOfInput):
            register[i] = 2 * np.random.random_sample() - 1 # random float [-1, 1)
        # initialize constant
        j = self.numberOfVariable + self.numberOfInput
        while j < register_length:
            register[j] = np.around(np.random.choice(np.arange(self.constInitRange[0],
                                self.constInitRange[1], self.constInitRange[2])), 2) #j - self.numberOfVariable + self.numberOfInput + 1
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
                                      self.isRandomSampling, self.evolutionStrategy)
        self.bestProgStr_ = bestProg.toString(self.numberOfVariable, self.numberOfInput, self.register_)
        effProg = copy.deepcopy(bestProg)
        effProg = effProg.eliminateStrcIntron()
        self.bestEffProgStr_ = effProg.toString(self.numberOfVariable, self.numberOfInput, self.register_)
        self.bestProg_ = bestProg
        self.bestProFitness_ = round(bestProg.fitness, 4)
        self.populationAvg_ = round(e.p.getAverageFitness(), 4)

        return self

    def predict(self, X):
        """
        Predict using the best fit genetic model.

        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        X = check_array(X)
        check_is_fitted(self)
        classType = self.classes_[0].dtype
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=classType)
        for i, row in enumerate(X):
            pred = self.bestProg_.predictProbaSigmoid(self.numberOfVariable, self.register_, row)
            if pred <= 0.5:  # class 0
                y_pred[i] = self.classes_[0]
            else:  # class 1
                y_pred[i] = self.classes_[1]
        return y_pred

    # def predict_proba(self, X):
    #     """
    #     Probability estimates.
    #     The returned estimates for all classes are ordered by the
    #     label of classes.
    #     Parameters
    #     ----------
    #     X : array-like of shape (n_samples, n_features)
    #        Vector to be scored, where `n_samples` is the number of samples and
    #        `n_features` is the number of features.
    #     Returns
    #     -------
    #     T : array-like of shape (n_samples, n_classes)
    #        Returns the probability of the sample for each class in the model,
    #        where classes are ordered as they are in ``self.classes_``.
    #     """
    #     check_is_fitted(self)
    #
    #     n_samples = X.shape[0]
    #     y_pred = np.zeros(n_samples, dtype=np.float64)
    #     for i, row in enumerate(X):
    #         y_pred[i] = self.bestProg_.predictProbaSigmoid(self.numberOfVariable, self.register_, row)
    #     return y_pred

    def _more_tags(self):
        return {'binary_only': True}

