import unittest
import numpy as np
from linear_genetic_programming._instruction import Instruction
from linear_genetic_programming._program import Program
from linear_genetic_programming._evolve import Evolve
from linear_genetic_programming._population import Population
from DataPreprocessing import DataPreprocessing
from linear_genetic_programming.lgp_classifier import LGPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
numberOfInput = 4
numberOfOperation = 5
numberOfVariable = 4
numberOfConstant = 9
macro_mutate_rate = 0.5
max_prog_ini_length = 15
min_prog_ini_length = 10
maxProgLength = 200
minProgLength = 10
pCrossover = 0.75
pConst = 0.5
pInsert = 0.5
pRegMut = 0.1
pMacro = 0.5
pMicro = 0.25
tournamentSize = 2
maxGeneration = 50
fitnessThreshold = 1.0
populationSize = 100

fitnessThreshhold = 1.0


#   register numberOfInput + numberOfVariable + numberOfConstant
def generateRegister():
    register_length = numberOfInput + numberOfVariable + numberOfConstant
    register = np.zeros(register_length, dtype=float)
    for i in range(numberOfVariable+numberOfInput):
        register[i] = 1
    j = numberOfVariable+numberOfInput
    while j < register_length:
        register[j] = j - numberOfVariable+numberOfInput + 1
        j += 1
    return register

register = generateRegister()


class Test_instruction(unittest.TestCase):

    def test_instruction_init(self):
        for _ in range(100):
            instruction = Instruction(numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant, pConst)
            print(instruction.toString(numberOfVariable, numberOfInput, register))

    def test_program_init(self):
        proLength = min_prog_ini_length + np.random.randint(max_prog_ini_length - minProgLength + 1)
        pro1 = Program()
        pro1.makeRandomeProg(numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant, proLength, pConst)
        print(pro1.toString(numberOfVariable, numberOfInput, register))
        return pro1

    def test_population_init(self):
        X_train, X_test, y_train, y_test = self.test_readDataIris()
        p = Population()
        p.generatePopulation(numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant,
                             pConst, max_prog_ini_length, min_prog_ini_length, populationSize)
        p.evaluatePopulation(numberOfVariable, register, X_train, y_train)
        # p.displayPopulationFitness()
        return p

    def test_micromutation(self):
        pro1 = self.test_program_init()
        p = self.test_population_init()
        test_e = Evolve(2, 100, p)  # tournamentSize, maxGeneration, population
        test_e.microMutation(pro1, pRegMut, pConst, numberOfVariable,
                      numberOfInput, numberOfOperation, numberOfConstant)
        print(pro1.toString(numberOfVariable, numberOfInput, register))

    def test_intronElimination(self):
        pro1 = self.test_program_init()
        #print(pro1.toString(numberOfVariable, numberOfInput, register))
        pro2 = pro1.eliminateStrcIntron()
        print(pro2.toString(numberOfVariable, numberOfInput, register))
        return True

    def test_evaluatePopulation(self):
        X_train, X_test, y_train, y_test = self.test_readDataIris()
        p = Population()
        p.generatePopulation(numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant,
                             pConst, max_prog_ini_length, min_prog_ini_length, populationSize)
        p.evaluatePopulation(numberOfVariable, register, X_train, y_train)
        p.displayPopulationFitness()
        print("average:")
        print(p.getAverageFitness())
        best = p.getBestIndividual()
        print(best.toString(numberOfVariable, numberOfInput, register))


    def test_readDataRuijin(self):
        path = "C:/Users/10455/Desktop/QueensLocal/CISC499/GeneticAlgorithm/LGP_PY/dataset/RuiJin_Processed.csv"
        X, y, names = DataPreprocessing.readDataRuiJin(path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test

    def test_readDataIris(self):
        # 0: "Versicolor", 1: "Virginica"
        path = "C:/Users/10455/Desktop/QueensLocal/CISC499/GeneticAlgorithm/LGP_PY/dataset/iris.csv"
        X, y, names= DataPreprocessing.readDataIris(path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test

    def test_lgpclassifierIris(self):
        X_train, X_test, y_train, y_test = self.test_readDataIris()
        # print(X_train.shape[1])
        lgp = LGPClassifier(numberOfInput = X_train.shape[1], numberOfVariable = 4, populationSize = 100,
                            fitnessThreshold = 1.0, maxGeneration = 20, showGenerationStat = True, tournamentSize=16,
                            randomSampling=False, evolutionStrategy="steady state")

        lgp.fit(X_train, y_train)
        # print(lgp.predict(X_test))
        print("Testing set accuracy")
        y_pred = lgp.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(lgp.bestProgStr_)
        print(lgp.bestEffProgStr_)

        #check_estimator(lgp)

    def test_lgpclassifierRuijin(self):
        X_train, X_test, y_train, y_test = self.test_readDataRuijin()
        #print(X_train.shape[1])
        lgp = LGPClassifier(numberOfInput = X_train.shape[1], numberOfVariable = 200, populationSize = 500,
                            fitnessThreshold = 0.95, max_prog_ini_length = 30, min_prog_ini_length = 10,
                            maxGeneration = 200, tournamentSize=16, evolutionStrategy = "steady state",
                            randomSampling=False, maxProgLength = 500)
        lgp.fit(X_train, y_train)
        y_pred = lgp.predict(X_test)
        print("Testing set accuracy")
        print(accuracy_score(y_test, y_pred))
        print(lgp.bestProgStr_)
        print(lgp.bestEffProgStr_)
