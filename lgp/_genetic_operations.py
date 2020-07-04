import numpy as np

class GeneticOperations:
    '''
    GeneticOperations implements crossover and two types of mutation

    '''

    @staticmethod
    def simpleCrossover(pro1, pro2):
        fracStart1 = np.random.randint(len(pro1.seq))
        fracEnd1 = fracStart1 + np.random.randint(len(pro1.seq ) -fracStart1)
        fracStart2 = np.random.randint(len(pro2.seq))
        fracEnd2 = fracStart2 + np.random.randint(len(pro2.seq) - fracStart2)

        frag1 = []
        frag2 = []
        for _ in range(fracStart1, fracEnd1):
            frag1.append(pro1.seq.pop(fracStart1))
        for _ in range(fracStart2, fracEnd2):
            frag2.append(pro2.seq.pop(fracStart2))

        while frag2:
            pro1.seq.insert(fracStart1, frag2.pop(0))
            fracStart1 += 1
        while frag1:
            pro2.seq.insert(fracStart2, frag1.pop(0))
            fracStart2 += 1
        return pro1, pro2

    @staticmethod
    def macroMutation(prog, pInsert, maxProgLength, minProgLength, randomInstr):
        choose = np.random.random_sample()
        if (len(prog.seq) < maxProgLength) and ((choose < pInsert) or len(prog.seq) == minProgLength):
            insertPos = np.random.randint(len(prog.seq))
            prog.seq.insert(insertPos, randomInstr)
        elif (len(prog.seq) > minProgLength) and ((choose >= pInsert) or len(prog.seq) == maxProgLength):
            deletePos = np.random.randint(len(prog.seq))
            del prog.seq[deletePos]

    @staticmethod
    def microMutation(prog, pRegMut, pConst, numberOfVariable,
                      numberOfInput, numberOfOperation, numberOfConstant):
        mutPos = np.random.randint(len(prog.seq))
        mutInstr = prog.seq[mutPos]

        if prog.seq[mutPos].isBranch:   # mutation of branch
            # print("branch mutation")
            regIndex = np.random.randint(1 ,3) # 1 or 2
            if regIndex == 1:
                GeneticOperations.__mutateRegister1(mutInstr, numberOfVariable, numberOfInput, numberOfConstant, pConst)
            elif regIndex == 2:
                GeneticOperations.__mutateRegister2(mutInstr, numberOfVariable, numberOfInput, numberOfConstant, pConst)
        # calculation mutation
        else:
            if np.random.random_sample() < pRegMut:  # register mutation
                regIndex = np.random.randint(3) # 3 register mutation situations

                if regIndex == 0:  # mutate return register
                    # print("mutate return register")
                    rt = np.random.randint(numberOfVariable)
                    while mutInstr.returnRegIndex == rt:
                        rt = np.random.randint(numberOfVariable)
                    mutInstr.returnRegIndex = rt
                elif regIndex == 1: # mutate register1
                    # print("mutate register1")
                    GeneticOperations.__mutateRegister1(mutInstr, numberOfVariable, numberOfInput, numberOfConstant, pConst)
                elif regIndex == 2:  # mutate register 2
                    GeneticOperations.__mutateRegister2(mutInstr, numberOfVariable, numberOfInput, numberOfConstant, pConst)
            else:  # operator mutation
                # print("operator mutation")
                index = mutInstr.operIndex
                while mutInstr.operIndex == index:
                    index = np.random.randint(numberOfOperation)
                mutInstr.operIndex = index

    @staticmethod
    def __mutateRegister1(mutInstr, numberOfVariable, numberOfInput, numberOfConstant, pConst):
        # print("mutate register1")
        index = mutInstr.reg1Index
        if mutInstr.reg2Index < numberOfVariable + numberOfInput:  # reg2 is a variable or input
            flip = np.random.random_sample()
            if flip >= pConst:  # reg1 will be a variable
                while mutInstr.reg1Index == index:
                    index = np.random.randint(numberOfVariable + numberOfInput)
                mutInstr.reg1Index = index
            else:  # reg1 will be a constant
                while mutInstr.reg1Index == index:
                    index = np.random.randint(numberOfConstant)
                mutInstr.reg1Index = numberOfVariable + numberOfInput + index
        else:  # reg2 is a constant then reg1 must be a variable
            while mutInstr.reg1Index == index:
                index = np.random.randint(numberOfVariable + numberOfInput)
            mutInstr.reg1Index = index

    @staticmethod
    def __mutateRegister2(mutInstr, numberOfVariable, numberOfInput, numberOfConstant, pConst):
        # print("mutate register 2")
        index = mutInstr.reg2Index
        if mutInstr.reg1Index < numberOfVariable + numberOfInput:  # reg1 is a variable or input
            flip = np.random.random_sample()
            if flip >= pConst:  # reg1 will be a variable
                while mutInstr.reg2Index == index:
                    index = np.random.randint(numberOfVariable + numberOfInput)
                mutInstr.reg2Index = index
            else:  # reg1 will be a constant
                while mutInstr.reg2Index == index:
                    index = np.random.randint(numberOfConstant)
                mutInstr.reg2Index = numberOfVariable + numberOfInput + index
        else:  # reg2 is a constant then reg1 must be a variable
            while mutInstr.reg2Index == index:
                index = np.random.randint(numberOfVariable + numberOfInput)
            mutInstr.reg2Index = index