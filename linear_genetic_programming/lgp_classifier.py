from linear_genetic_programming._evolve import Evolve
from linear_genetic_programming._population import Population
import numpy as np
import copy
import pickle

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score


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
        as part of a program. Number of variable size should be at least half of feature size.

    numberOfConstant: integer, optional, (default=9)
        Number of constant in register. Constants are stored in registers that are
        write-protected. Constant registers are only initialized once at the beginning
        with values from a constInitRange.

    max_prog_ini_length: integer, optional, (default=30)
        Max program initialization length

    min_prog_ini_length: integer, optional, (default=10)
        Min program initialization length

    maxProgLength: integer, optional, (default=300)
        maximum program length limit during evolution.

    minProgLength: integer, optional, (default=10)
        minimum program length required during evolution.

    pCrossover: float, optional, (default=0.75)
        Probability of exchanging the genetic information of two parent programs

    pConst: float, optional, (default=0.5)
        Control the probability of constant in Instruciton initialization. It controls whether the
        register will be a constant. It also controls mutation probability in micromutaion. It controls
        whether a register will mutate to constant.

    pInsert: float, optional, (default=0.5)
        Control probability of insertion in macromutation. It will insert a random instruction
        into the program.

    pRegmut: float, optional, (default=0.6)
        Control probability of register mutation used in micromutaion. It will either mutate
        register1, register2 or return register.

    pMacro: float, optional, (default=0.75)
        Probability of macromutation, Macromutation operate on the level of program. It will add
        or delete instruction. It will affect program size.

    pMicro: float, optional, (default=0.5)
        Probability of micromuation. Micromuation operate on the level of instruction components
        (micro level) and manipulate registers, operators, and constants.

    tournament_size : integer, optional, (default=2)
        The size of tournament selection. The number of programs that will compete to become part of the next
        generation.

    maxGenerations : integer, optional, (default=200)
        The number of generations to evolve.

    fitnessThreshold: float, optional, (default=1.0)
        When not using random sampling, terminate the evolution if threshold is met.
        When using random sampling, fitnessThreshold has no effect.

    populationSize: integer, optional, (default=1000)
        Size of population

    showGenerationStat: boolean, optional, (default=True)
        If True, print out statistic in each generation.
        Set to False to save time. Some average statistical calculations is time consuming.

    isRandomSampling: Boolean, optional, (default=True)
        Train the genetic algorithm on random sampled dataset (without replacement)

    constInitRange: tuple (start, stop, step), optional, (default=(1,11,1))
        Initiation of the constant set. range: [start, stop).

    randomState: int, default=None
        Controls both the randomness of the algorithm.

    testingAccuracy: int
        used to save testing set accuracy score

    validationScores: dict
        used to hold validation metrics during running

    names: list
        feature names of the dataset

    Attributes
    ----------
    register_: array of shape (numberOfInput + numberOfVariable + numberOfConstant, )
        Register stores the calculation variables, feature values and constants.

    bestProg_: class Program
        A list of Instructions used for classification calculation

    bestEffProg_:
        Best program with struct intron and semantic intron removed

    bestProFitness_ : float
        Training set accuracy score of the best program

    bestProgStr_: str
        String representation of the best program

    bestEffProgStr_: str
        Intron removed program string representation

    populationAvg_: float
        Average fitness of the final generation

    '''

    def __init__(self,
                 numberOfInput,
                 numberOfOperation=5,
                 numberOfVariable=4,
                 numberOfConstant=9,
                 max_prog_ini_length=30,
                 min_prog_ini_length=10,
                 maxProgLength=300,
                 minProgLength=10,
                 pCrossover=0.75,
                 pConst=0.5,
                 pInsert=0.5,
                 pRegmut=0.6,
                 pMacro=0.75,
                 pMicro=0.5,
                 tournamentSize=2,
                 maxGeneration=200,
                 fitnessThreshold=1.0,
                 populationSize=1000,
                 showGenerationStat=True,
                 isRandomSampling=True,
                 constInitRange=(1, 11, 1),
                 randomState=None,
                 testingAccuracy=-1,
                 validationScores=None,
                 names=None):
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
        self.constInitRange = constInitRange
        self.randomState = randomState
        self.testingAccuracy = testingAccuracy
        self.validationScores = validationScores
        self.names = names

    def __generateRegister(self):
        # Initialization of register
        # register numberOfInput + numberOfVariable + numberOfConstant
        register_length = self.numberOfInput + self.numberOfVariable + self.numberOfConstant
        register = np.zeros(register_length, dtype=float)
        for i in range(self.numberOfVariable + self.numberOfInput):
            register[i] = 2 * np.random.random_sample() - 1  # random float [-1, 1)
        # initialize constant
        j = self.numberOfVariable + self.numberOfInput
        while j < register_length:
            register[j] = np.around(np.random.choice(np.arange(self.constInitRange[0],
                                                               self.constInitRange[1], self.constInitRange[2])),
                                    2)  # j - self.numberOfVariable + self.numberOfInput + 1
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
        if self.randomState is not None:
            np.random.seed(self.randomState)

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
                                      self.isRandomSampling)
        self.bestProgStr_ = bestProg.toString(self.numberOfVariable, self.numberOfInput, self.register_)
        effProg = copy.deepcopy(bestProg)
        effProg = effProg.eliminateStrcIntron()
        effProg = self.__remove_semantic_intron(X, y, effProg)
        self.bestEffProg_ = effProg
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
        # check_is_fitted(self)
        classType = self.classes_[0].dtype
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=classType)
        for i, row in enumerate(X):
            y_pred[i] = self.bestProg_.predictProbaSigmoid(self.numberOfVariable, self.register_, row, self.classes_)
        return y_pred

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
        # check_is_fitted(self)

        n_samples = X.shape[0]
        # only support binary classification
        y_pred = np.zeros((n_samples, 2), dtype=np.float64)
        for i, row in enumerate(X):
            sigmoid_pred = self.bestProg_.predictProbaSigmoid(self.numberOfVariable, self.register_, row,
                                                              self.classes_, 'prob')
            y_pred[i, 0] = 1 - sigmoid_pred  # when <= 0.5, there is larger probability in class 0
            y_pred[i, 1] = sigmoid_pred  # when >0.5, there is larger probability in class 1
        return y_pred

    def save_model(self, fname='lgp.pkl', mode='ab'):
        '''
        Save the current object into a pickle file. Assuming the file is in the
        same directory.

        Parameters
        ----------
        fname: string (default = 'lgp.pkl')
            file name of the output

        Returns
        -------
        True:
            if successfully saved

        '''
        with open(fname, mode) as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        return True

    @classmethod
    def load_model(cls, fname='lgp.pkl', mode="rb"):
        '''
        load lgp object from a pickle file. Assuming the file is in the
        same directory

        Parameters
        ----------
        fname: string (default = 'lgp.pkl')
            file name of the output

        Returns
        -------
        lgp: LGPClassifier generator
            generator
        '''
        with open(fname, mode) as input_f:
            while True:
                try:
                    yield pickle.load(input_f, encoding='bytes')
                except EOFError:
                    break

    @classmethod
    def load_model_directly(cls, pickle_file_input):
        '''
        Used to read a file in website

        Parameters
        ----------
        pickle_file_input: byte stream
            BytesIO input

        Returns
        -------
        lgp: LGPClassifier generator
            generator
        '''
        while True:
            try:
                yield pickle.load(pickle_file_input, encoding='bytes')
            except EOFError:
                break

    # semantic intron does not alter the value stored in r0
    # Algorithm 3.1 detection of structural introns from LGP book
    def __remove_semantic_intron(self, X, y, Prog):
        remove_index = []
        p = copy.deepcopy(Prog)
        for delete_index, _ in enumerate(p.seq):  # try to delete one instruction a time
            y_pred1 = np.zeros(X.shape[0], dtype=self.classes_[0].dtype)
            for i, row in enumerate(X):
                y_pred1[i] = p.predictProbaSigmoid(self.numberOfVariable, self.register_, row,
                                                   self.classes_)
            result1_acc = accuracy_score(y, y_pred1)

            p2 = copy.deepcopy(p)
            del p2.seq[delete_index]  # try to delete instruction
            y_pred2 = np.zeros(X.shape[0], dtype=self.classes_[0].dtype)
            for i, row in enumerate(X):
                y_pred2[i] = p2.predictProbaSigmoid(self.numberOfVariable, self.register_, row,
                                                    self.classes_)
            result2_acc = accuracy_score(y, y_pred2)

            if np.array_equal(y_pred1, y_pred2) and result1_acc == result2_acc:  # result is the same
                remove_index.append(delete_index)
        for index in sorted(remove_index, reverse=True):
            del p.seq[index]
        return p

    def _more_tags(self):
        return {'binary_only': True}
