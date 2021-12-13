import numpy as np


class Instruction:
    """
    Instruction is the lowest level in classification model. It contains a return register,
    an operation and two calculation registers. For example, "r[0] = r[2] + 0.5", r[0] is
    a return register, r[2] and 0.5 are calculation registers. '+' is operation.

    Parameters
    ----------
    numberOfOperation
    numberOfVariable
    numberOfInput
    numberOfConstant
    pConst, pBranch

    Attributes
    ----------
    isBranch
    branchType
    operIndex
        calculated using random number in len(numberOfOperation)
    returnRegIndex
        calculated using random number in len(numberOfVariable)
    reg1Index
    reg2Index

    """

    # np.random.seed(0)
    OP_ADD = 0
    OP_SUBTRACT = 1
    OP_MULTIPLY = 2
    OP_DIVIDE = 3
    OP_EXPON = 4

    def __init__(self, numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant, pConst, pBranch=0.3):
        self.isBranch = np.random.random_sample() < pBranch
        if self.isBranch:  # branch instruction
            self.branchType, self.reg1Index, self.reg2Index = \
                self.makeRandomInstr(numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant, pConst)
        else:
            self.operIndex, self.returnRegIndex, self.reg1Index, self.reg2Index = \
                self.makeRandomInstr(numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant, pConst)

    def makeRandomInstr(self, numberOfOperation, numberOfVariable, numberOfInput, numberOfConstant, pConst):
        """ make a random instruction """
        if np.random.random() >= pConst:  # reg1 will be a variable or input
            r1 = np.random.randint(numberOfVariable + numberOfInput)
            reg1Index = r1
            if np.random.random() >= pConst:  # reg2 will be a variable or input
                r2 = np.random.randint(numberOfVariable + numberOfInput)
                reg2Index = r2
            else:  # reg2 will be a constant
                r2 = np.random.randint(numberOfConstant)
                reg2Index = numberOfVariable + numberOfInput + r2
        else:  # reg1 will be a constant and reg2 will be a variable or input
            r1 = np.random.randint(numberOfConstant)
            reg1Index = numberOfVariable + numberOfInput + r1
            r2 = np.random.randint(numberOfVariable + numberOfInput)
            reg2Index = r2
        if self.isBranch:
            branch_type = ["if less", "if greater"]
            return np.random.choice(branch_type), reg1Index, reg2Index
        else:
            operIndex = np.random.randint(numberOfOperation)
            # since zero is return register in calculation, make sure there are enough zeros by increasing its chance
            pZero = 0.0004 * numberOfInput
            returnRegIndex = 0 if pZero > np.random.random_sample() else np.random.randint(numberOfVariable)
            return operIndex, returnRegIndex, reg1Index, reg2Index

    def toString(self, numberOfVariable, numberOfInput, register):
        """ Convert a program to a readable format """
        s = "<"

        if self.isBranch:
            s += self.branchType
            s += ", "
        else:
            if self.operIndex == self.OP_ADD:
                s += "+"
            elif self.operIndex == self.OP_MULTIPLY:
                s += "*"
            elif self.operIndex == self.OP_SUBTRACT:
                s += "-"
            elif self.operIndex == self.OP_DIVIDE:
                s += "/"
            elif self.operIndex == self.OP_EXPON:
                s += "^"
            s += ", r" + str(self.returnRegIndex) + ", "
        if self.reg1Index >= numberOfVariable + numberOfInput:  # It is a constant
            s += str(register[self.reg1Index])
        else:
            s += "r" + str(self.reg1Index)
        s += ", "
        if self.reg2Index >= numberOfVariable + numberOfInput:  # It is a constant
            s += str(register[self.reg2Index])
        else:
            s += "r" + str(self.reg2Index)

        s += ">"
        return s
