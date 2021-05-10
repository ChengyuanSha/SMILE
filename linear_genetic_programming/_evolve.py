from linear_genetic_programming._instruction import Instruction
from linear_genetic_programming._genetic_operations import GeneticOperations
import numpy as np
import copy


class Evolve:
    '''
    A population of diverse candidate prediction models evolve to improve prediction
    accuracy gradually through a number of generations. After evolution halts, the best
    model of the population in the final generation will be the output.
    '''

    # np.random.seed(0)

    def __init__(self, tournamentSize, maxGeneration, population):
        self.tournamentSize = tournamentSize
        self.maxGeneration = maxGeneration
        self.p = population

    def twoTournamentSelectionReturnIndex(self):
        '''
        Perform two tournament selection without replacement.

        Returns
        -------
        winner1 index
        winner2 index
        loser1 index
        loser2 index

        '''
        if self.tournamentSize*2 > len(self.p.population):
            raise ValueError("2 * Tournament size cannot be larger than population")
        winner1, winner2, loser1, loser2 = None, None, None, None
        win1Idx, win2Idx, loser1dx, loser2dx = 0, 0, 0, 0
        rand1, rand2 = np.split(np.random.choice(len(self.p.population), self.tournamentSize*2, replace=False), 2)
        for i in range(self.tournamentSize):
            cur = self.p.population[rand1[i]]
            if winner1 == None or cur.fitness > winner1.fitness:
                winner1 = cur
                win1Idx = rand1[i]
            if loser1 == None or cur.fitness < loser1.fitness:
                loser1 = cur
                loser1dx = rand1[i]
        for i in range(self.tournamentSize):
            cur = self.p.population[rand2[i]]
            if winner2 == None or cur.fitness > winner2.fitness:
                winner2 = cur
                win2Idx = rand2[i]
            if loser2 == None or cur.fitness < loser2.fitness:
                loser2 = cur
                loser2dx = rand2[i]
        return win1Idx, win2Idx, loser1dx, loser2dx

    @staticmethod
    def displayHeader():
        print("{:^3}|{:^9}|{:^6}|{:^7}|{:^12}|{:^10}|{:^13}".format("Gen", "Best Indv", "CE", "Pop Avg", "Ran Sampling", "AvgProgLen", "AvgEffProgLen"))
        print('-' * 3 + ' ' + '-' * 9 + ' ' + '-' * 6 + ' ' + '-' * 7 + ' ' + '-' * 12 + ' ' + '-' * 10 + ' ' + '-' * 13)

    def displayStatistics(self, g, bestIndividual, isRandomSampling, randomSz):
        ranSamplingStatus = randomSz if isRandomSampling else -1
        line_format = '{0:>3d}|{1:>9.2f}|{2:>6d}|{3:>7.2f}|{4:>12d}|{5:>10.2f}|{6:>13.2f}'
        print(line_format.format(g, bestIndividual.fitness, bestIndividual.classificationError,
              self.p.getAverageFitness(), ranSamplingStatus, self.p.getAvgProgLen(), self.p.getAvgEffProgLen() ))

    def evolveGeneration(self, pRegMut, pMicro, pMacro, pConst, pCrossover, numberOfVariable, numberOfInput, numberOfOperation,
                         numberOfConstant, register, pInsert, maxProgLength, minProgLength,
                         X_train, y_train, fitnessThreshold, showGenerationStat, isRandomSampling):
        '''
        Implements "Linear genetic programming" Algorithm 2.1
        '''
        samplingSize = -1
        if showGenerationStat:
            Evolve.displayHeader()
        self.p.evaluatePopulation(numberOfVariable, register, X_train, y_train)
        bestIndividual = self.p.getBestIndividual()

        for g in range(self.maxGeneration):

            for _ in range(len(self.p.population)):
                # Perform two fitness tournaments
                win1Idx, win2Idx, loser1Idx, loser2Idx = self.twoTournamentSelectionReturnIndex()

                winner1 = copy.deepcopy(self.p.population[win1Idx])
                winner2 = copy.deepcopy(self.p.population[win2Idx])

                # Modify the two winners by one or more variation operators with certain probabilities(cross over)
                if np.random.random_sample() < pCrossover:
                    winner1, winner2 = GeneticOperations.simpleCrossover(winner1, winner2)
                # Mutation
                if np.random.random_sample() < pMacro:
                    randomInstr = Instruction(numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant, pConst)
                    GeneticOperations.macroMutation(winner1, pInsert, maxProgLength, minProgLength, randomInstr)
                if np.random.random_sample() < pMacro:
                    randomInstr = Instruction(numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant, pConst)
                    GeneticOperations.macroMutation(winner2, pInsert, maxProgLength, minProgLength, randomInstr)
                if np.random.random_sample() < pMicro:
                    GeneticOperations.microMutation(winner1, pRegMut, pConst, numberOfVariable, numberOfInput, numberOfOperation,
                                       numberOfConstant)
                if np.random.random_sample() < pMicro:
                    GeneticOperations.microMutation(winner2, pRegMut, pConst, numberOfVariable, numberOfInput, numberOfOperation,
                                       numberOfConstant)

                # Random Sampling Technique
                if isRandomSampling:
                    # random sampling in constant size, half of the population
                    X_size = X_train.shape[0]
                    samplingSize = np.random.choice(np.arange(X_size//2, X_size))
                    indexArray = np.random.choice(X_size, samplingSize, replace=False)
                    X_subset = X_train[indexArray, :]
                    y_subset = y_train[indexArray]
                    winner1.evaluate(numberOfVariable, register, X_subset, y_subset)
                    winner2.evaluate(numberOfVariable, register, X_subset, y_subset)
                else:
                    winner1.evaluate(numberOfVariable, register, X_train, y_train)
                    winner2.evaluate(numberOfVariable, register, X_train, y_train)

                if winner1.fitness > bestIndividual.fitness:
                    bestIndividual = winner1
                if winner2.fitness > bestIndividual.fitness:
                    bestIndividual = winner2

                self.p.population[loser1Idx] = winner1
                self.p.population[loser2Idx] = winner2

            if showGenerationStat:
                self.displayStatistics(g, bestIndividual, isRandomSampling, samplingSize)

            if not isRandomSampling:
            # if the solution has been found, end the loop
                if bestIndividual.fitness >= fitnessThreshold:
                    return bestIndividual

        return bestIndividual
