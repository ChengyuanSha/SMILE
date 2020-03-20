from linear_genetic_programming._instruction import Instruction
import copy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels

class Program:
    '''
    A list of instructions

    Attributes
    ----------
    seq : python list
        contain a list of Instructions
    fitness: int
        set after evaluate function, contain calculated fitness
    classificationError: int
        set after evaluate function, number of misclassified samples
    effProgLen: int
        set after evaluate function, store effective program length
    '''
    OP_ADD = 0
    OP_SUBTRACT = 1
    OP_MULTIPLY = 2
    OP_DIVIDE = 3
    OP_EXPON = 4

    def __init__(self):
        self.seq = []
        # self.fitness = -1
        # self.classificationError = -1

    def makeRandomeProg(self, numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant, length, pConst):
        self.seq = [Instruction(numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant, pConst, 0.5) for _
                    in range(length - 1)]
        # make sure last instruction is not a branch
        self.seq.append(Instruction(numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant, pConst, 0))

    def toString(self, numberOfVariable, numberOfInput, register):
        s = ""
        count = 0
        if self.seq == []:
            return "empty program"
        for i in self.seq:
            s += "I" + str(count) + ":  " + i.toString(numberOfVariable, numberOfInput, register) + "\n"
            count += 1
        return s

    # input is 1 dimension 1 row data
    def execute(self, numberOfVariable, register, X_train):
        data_type = X_train[0].dtype
        check_float_range = lambda x: np.clip(x , -np.sqrt(np.finfo(data_type).max), np.sqrt(np.finfo(data_type).max))
        registerCopy = copy.deepcopy(register)
        for i in range(len(X_train)):
            registerCopy[i + numberOfVariable] = X_train[i]
        i = 0
        while i < len(self.seq):
            if self.seq[i].isBranch:  # branch Instruction
                if self.seq[i].branchType == "if less":
                    branch_result = registerCopy[self.seq[i].reg1Index] < registerCopy[self.seq[i].reg2Index]
                else:  # "if greater":
                    branch_result = registerCopy[self.seq[i].reg1Index] > registerCopy[self.seq[i].reg2Index]
                if not (branch_result):  # if branch false, one instruction is skipped
                    while i < len(self.seq) - 1 and self.seq[i + 1].isBranch:  # if next is still a branch
                        i += 1
                    i += 2
                else:  # if branch true, execute next instruction
                    i += 1
            else:  # not a branch Instruction
                if self.seq[i].operIndex == self.OP_ADD:
                    registerCopy[self.seq[i].returnRegIndex] = check_float_range(registerCopy[self.seq[i].reg1Index]) \
                        + check_float_range(registerCopy[self.seq[i].reg2Index])
                elif self.seq[i].operIndex == self.OP_SUBTRACT:
                    registerCopy[self.seq[i].returnRegIndex] = check_float_range(registerCopy[self.seq[i].reg1Index]) \
                        - check_float_range(registerCopy[self.seq[i].reg2Index])
                elif self.seq[i].operIndex == self.OP_MULTIPLY:
                    registerCopy[self.seq[i].returnRegIndex] = check_float_range(registerCopy[self.seq[i].reg1Index]) \
                        * check_float_range(registerCopy[self.seq[i].reg2Index])
                elif self.seq[i].operIndex == self.OP_DIVIDE:  # protected operation
                    if registerCopy[self.seq[i].reg2Index] != 0:
                        registerCopy[self.seq[i].returnRegIndex] = check_float_range(registerCopy[self.seq[i].reg1Index]) \
                            / check_float_range(registerCopy[self.seq[i].reg2Index])
                    else:
                        registerCopy[self.seq[i].returnRegIndex] = registerCopy[self.seq[i].reg1Index]
                elif self.seq[i].operIndex == self.OP_EXPON:  # protected operation
                    if np.abs(registerCopy[self.seq[i].reg2Index]) <= 10 and np.abs(registerCopy[self.seq[i].reg1Index]) != 0:
                        registerCopy[self.seq[i].returnRegIndex] = np.float_power(
                            np.abs(registerCopy[self.seq[i].reg1Index]), registerCopy[self.seq[i].reg2Index])
                    else:
                        registerCopy[self.seq[i].returnRegIndex] = check_float_range(registerCopy[self.seq[i].reg1Index]) \
                            + check_float_range(registerCopy[self.seq[i].reg2Index])
                i += 1
        return registerCopy[0]

    @staticmethod
    def sigmoid(x):
        # Numerically stable sigmoid function.
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def evaluate(self, numberOfVariable, register, X, y):
        y_class = unique_labels(y)
        y_pred = np.zeros(len(y), dtype=y_class[0].dtype)
        for i, row in enumerate(X):
            y_pred[i] = self.predictProbaSigmoid(numberOfVariable, register, row, y_class)
        self.fitness = accuracy_score(y, y_pred)
        self.classificationError = np.argwhere(y != y_pred).size

    def predictProbaSigmoid(self, numberOfVariable, register, singleX, classes, returnType='class'):
        '''
        Parameters
        ----------
        numberOfVariable

        register
            register used in calculation
        singleX: array
            a training sample, row in X
        classes: array of size 2
            binary class type, eg. [0, 1]

        Returns
        -------
        I: integer
            return class 0 or class 1 based on the sigmoid function if returntype == 'class'
        P: float
            return probability if returntype == 'prob'
        '''
        self.progLen = len(self.seq)
        exonProgram = copy.deepcopy(self)
        exonProgram = exonProgram.eliminateStrcIntron()
        self.effProgLen = len(exonProgram.seq)
        result = exonProgram.execute(numberOfVariable, register, singleX)
        pred = Program.sigmoid(result)
        if returnType == 'class':
            if pred <= 0.5:  # class 0
                return classes[0]
            else:  # class 1
                return classes[1]
        elif returnType == 'prob':
            return pred

    def eliminateStrcIntron(self):
        strucIntronFreeProg = Program()
        effInstr = []
        effReg = []
        effReg.append(0)
        i = len(self.seq) - 1
        while i >= 0:
            if not (self.seq[i].isBranch):  # not a branch
                if self.seq[i].returnRegIndex in effReg:
                    if not (self.seq[i].reg1Index in effReg):
                        effReg.append(self.seq[i].reg1Index)
                    if not (self.seq[i].reg2Index in effReg):
                        effReg.append(self.seq[i].reg2Index)
                    if (self.seq[i].returnRegIndex != self.seq[i].reg1Index) and \
                            (self.seq[i].returnRegIndex != self.seq[i].reg2Index):  # irrelevant
                        effReg.remove(self.seq[i].returnRegIndex)
                    effInstr.insert(0, i)
                    # If the operation directly follows a branch or a sequence of branches then add these instructions
                    while (i - 1) >= 0 and self.seq[i - 1].isBranch:
                        effInstr.insert(0, i - 1)
                        i -= 1
            i -= 1
        for i in range(len(effInstr)):
            strucIntronFreeProg.seq.append(self.seq[effInstr[i]])
        return strucIntronFreeProg
