from linear_genetic_programming._program import Program
import numpy as np


class Population:
    """
    Population contains many programs, where each one is independent.

    Parameters
    ----------

    Attributes
    ----------
    population

    """

    def __init__(self):
        self.population = []

    def generatePopulation(self, numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant,
                           pConst, max_prog_ini_length, min_prog_ini_length, populationSize):
        """ Initialize a population """
        for _ in range(populationSize):
            proLength = min_prog_ini_length + np.random.randint(max_prog_ini_length - min_prog_ini_length + 1)
            pro = Program()
            pro.makeRandomeProg(numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant, proLength, pConst)
            self.population.append(pro)

    def evaluatePopulation(self, numberOfVariable, register, X_train, y_train):
        """ Evaluate fitness of the population """
        for i in self.population:
            i.evaluate(numberOfVariable, register, X_train, y_train)

    def displayPopulationFitness(self):
        """ Display fitness values in population """
        for i, ele in enumerate(self.population):
            print("Program " + str(i) + " fitness:" + str(ele.fitness))

    def getBestIndividual(self):
        """ Return the best program """
        return max(self.population, key=lambda x: x.fitness)

    def getAverageFitness(self):
        """ Return average fitness of the population """
        return np.mean([prog.fitness for prog in self.population])

    def getAvgEffProgLen(self):
        """ Return effective length statistic """
        return np.mean([prog.effProgLen for prog in self.population])

    def getAvgProgLen(self):
        """ Return average program length statistic """
        return np.mean([prog.progLen for prog in self.population])
