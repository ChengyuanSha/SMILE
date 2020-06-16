import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from linear_genetic_programming.lgp_classifier import LGPClassifier
from collections import Counter
import re
import copy
import itertools
from scipy.sparse import csr_matrix

class ResultProcessing:
    '''
    util class

    Attributes
    ----------
    X
    y
    names
    model_list
    sample_list
    feature_list
        numpy array
    calculation_variable_list
    '''
    def __init__(self, original_data_file_path):
        self.original_data_file_path = original_data_file_path
        self.X, self.y, self.names = self.readDataRuiJinAD()

    def get_original_dataframe(self):
        return pd.read_csv(self.original_data_file_path)

    def readDataRuiJin(self):
        df = self.get_original_dataframe()
        names = df.columns[1:]
        # AMCI VS Normal
        df = df[(df['category'] == 2) | (df['category'] == 3)]
        y = df['category'].values.astype('int64')
        y = np.where(y == 2, 0, y)
        y = np.where(y == 3, 1, y)
        X = df.iloc[:, 1:].values
        scaler = MinMaxScaler((-1, 1))
        X = scaler.fit_transform(X)
        return X, y, names

    def readDataRuiJinAD(self):
        df = self.get_original_dataframe()
        names = df.columns[1:]
        # AD VS Normal
        df = df[(df['category'] == 1) | (df['category'] == 3)]
        y = df['category'].values.astype('int64')
        y = np.where(y == 1, 0, y)
        y = np.where(y == 3, 1, y)
        X = df.iloc[:, 1:].values
        scaler = MinMaxScaler((-1, 1))
        X = scaler.fit_transform(X)
        return X, y, names

    # load models
    def load_models_from_file_path(self, pickle_file_path):
        lgp_models = LGPClassifier.load_model(pickle_file_path)
        model_list = [i for i in lgp_models]
        self.model_list = model_list

    def load_models_directly(self, input):
        lgp_models = LGPClassifier.load_model_directly(input)
        model_list = [i for i in lgp_models]
        self.model_list = model_list

    # return feature list and calculation variable list
    def calculate_featureList_and_calcvariableList(self):
        numOfVariable = self.model_list[0].numberOfVariable
        feature_list = []
        program_length_list = []
        for i in self.model_list:
            feature_list.append(re.findall(r'r\d+', i.bestEffProgStr_))
            program_length_list.append(i.bestEffProgStr_.count('\n'))
        calculation_variable_list = copy.deepcopy(feature_list)  # raw list for later usage
        # processing raw list to get calculation_variable_list
        i = 0
        while i < len(calculation_variable_list):
            j = 0
            program = calculation_variable_list[i]
            while j < len(program):
                if int(calculation_variable_list[i][j][1:]) > numOfVariable:  # remove calculation variable
                    del calculation_variable_list[i][j]
                else:
                    calculation_variable_list[i][j] = int(calculation_variable_list[i][j][1:])
                    j += 1  # ONLY INCREMENT HERE
            i += 1
        i = 0
        # processing raw list to get feature_list
        while i < len(feature_list):
            j = 0
            program = feature_list[i]
            while j < len(program):
                if int(feature_list[i][j][1:]) < numOfVariable:  # remove calculation variable
                    del feature_list[i][j]
                else:
                    feature_list[i][j] = int(feature_list[i][j][1:]) - numOfVariable
                    j += 1  # ONLY INCREMENT HERE
            i += 1
        self.feature_list = np.asarray(feature_list)
        self.calculation_variable_list = calculation_variable_list

    def get_occurrence_from_feature_list_given_length(self, given_length):
        element = np.asarray([i for i in self.feature_list if len(i) == given_length])
        if len(element) == 0:
            raise ValueError("There is no program in this length")
        rank = Counter(element.flatten())
        features, num_of_occurrences = zip(*rank.most_common())
        return features, num_of_occurrences, len(element)

    def get_accuracy_given_length(self, given_length):
        prog_index = np.asarray([c for c, i in enumerate(self.feature_list) if len(i) == given_length])
        acc_scores = [ self.model_list[i].bestProFitness_ for i in prog_index ] # calculated in filter_model.py
        return prog_index, acc_scores

    # private method
    @staticmethod
    def __create_co_occurences_matrix(allowed_words, documents):
        word_to_id = dict(zip(allowed_words, range(len(allowed_words))))
        documents_as_ids = [np.sort([word_to_id[w] for w in doc if w in word_to_id]).astype('uint32') for doc in
                            documents]
        row_ind, col_ind = zip(*itertools.chain(*[[(i, w) for w in doc] for i, doc in enumerate(documents_as_ids)]))
        data = np.ones(len(row_ind), dtype='uint32')  # use unsigned int for better memory utilization
        max_word_id = max(itertools.chain(*documents_as_ids)) + 1
        docs_words_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(
                            len(documents_as_ids), max_word_id))  # efficient arithmetic operations with CSR * CSR
        cooc_matrix = docs_words_matrix.T * docs_words_matrix  # multiplying docs_words_matrix with its transpose matrix would generate the co-occurences matrix
        cooc_matrix.setdiag(0)
        return cooc_matrix, word_to_id

    def get_feature_co_occurences_matrix(self, given_length):
        # feature index
        index = np.asarray([c for c, i in enumerate(self.feature_list) if len(i) == given_length])
        # filter feature list
        document = self.feature_list[index]
        f, _, _ = self.get_occurrence_from_feature_list_given_length(given_length)
        feature_index = list(f)
        cooc_matrix, _ = ResultProcessing.__create_co_occurences_matrix(feature_index, document)
        cooc_matrix = cooc_matrix.todense()
        return cooc_matrix, feature_index

    def get_index_of_models_given_feature_and_length(self, feature_num, given_length):
        return [c for c, i in enumerate(self.feature_list) if len(i) == given_length and feature_num in i]

    def convert_program_str_repr(self, model):
        # convert the raw string to user friendly string
        s = ''
        original_str = model.bestEffProgStr_.splitlines()
        i = 0
        indentation = False
        indentation_level = 1
        while i < len(original_str):
            current_string = original_str[i]
            vars_in_line = re.findall(r'r\d+', current_string)
            # reformat structure
            if 'if' not in current_string:
                extract = [x.strip() for x in current_string.split(',')]
                current_string = extract[0][:3] + ' '+ extract[1] + ' = ' +  extract[2] + ' ' + extract[0][-1] + ' ' + extract[3][:-1]
            # substitute variable index
            for var in vars_in_line:
                var = var[1:]
                if int(var) < model.numberOfVariable and int(var) != 0: # calculation variable
                    current_string = re.sub('r' + re.escape(var), str(round(model.register_[int(var)], 2)),
                                            current_string)
                elif int(var) > model.numberOfVariable and int(var) != 0: # features
                    name_index = int(var) - model.numberOfVariable
                    current_string = re.sub('r' + re.escape(var), str(self.names[name_index]), current_string)
            # take care of indentation
            if indentation:
                current_string = current_string[:3] + indentation_level*'  ' + 'then ' + current_string[3:]
            if 'if' in current_string:
                indentation = True
                indentation_level += 1
            else:
                indentation = False
            s += current_string + '\n'
            i += 1
        s += 'Output register r[0] will then go through sigmoid transformation S \nif S(r[0]) is less or equal ' \
             'than 0.5:\n  this sample will be classified by this model as class 0, i.e. diseased. \nelse:\n' \
             '  class 1, i.e. not diseased'
        print(s)
        return s

if __name__ == '__main__':
    # some small testing code
    result = ResultProcessing("../dataset/RuiJin_Processed.csv")
    result.load_models_from_file_path("../dataset/lgp_acc.pkl")
    X, y, names = result.readDataRuiJinAD()
    result.calculate_featureList_and_calcvariableList()
    # prog_index, acc_scores =  result.get_accuracy_given_length(1)
    # index = result.get_index_of_models_given_feature_and_length(87, 1)
    # print(index)
    # for i in index:
    #     print(result.model_list[i].bestEffProgStr_)
    print(result.model_list[899].bestEffProgStr_)
    result.convert_program_str_repr(result.model_list[20])

