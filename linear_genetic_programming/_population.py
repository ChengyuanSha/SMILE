from linear_genetic_programming._program import Program
import numpy as np


class Population:
    '''
    Population is a list of programs
    '''

    def __init__(self):
        self.population = []

    def generatePopulation(self, numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant,
                           pConst, max_prog_ini_length, min_prog_ini_length, populationSize):
        for _ in range(populationSize):
            proLength = min_prog_ini_length + np.random.randint(max_prog_ini_length - min_prog_ini_length + 1)
            pro = Program()
            pro.makeRandomeProg(numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant, proLength, pConst)
            self.population.append(pro)

    def evaluatePopulation(self, numberOfVariable, register, X_train, y_train):
        for i in self.population:
            i.evaluate(numberOfVariable, register, X_train, y_train)

    def displayPopulationFitness(self):
        for i, ele in enumerate(self.population):
            print("Program " + str(i) + " fitness:" + str(ele.fitness))

    def getBestIndividual(self):
        return max(self.population, key= lambda x: x.fitness)

    def getAverageFitness(self):
        return np.mean([prog.fitness for prog in self.population])