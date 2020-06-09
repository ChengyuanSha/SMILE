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
    def __init__(self, original_data_file_path, pickle_file_path):
        self.original_data_file_path = original_data_file_path
        self.pickle_file_path = pickle_file_path
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

    # load and filter models, this function is too slow!!! need to fix!!!
    def load_models(self):
        lgp_models = LGPClassifier.load_model(self.pickle_file_path)
        model_list = [i for i in lgp_models][0]
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
        rank = Counter(element.flatten())
        features, num_of_occurrences = zip(*rank.most_common())
        return features, num_of_occurrences

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
        f, _ = self.get_occurrence_from_feature_list_given_length(given_length)
        feature_index = list(f)
        cooc_matrix, _ = ResultProcessing.__create_co_occurences_matrix(feature_index, document)
        cooc_matrix = cooc_matrix.todense()
        return cooc_matrix, feature_index

if __name__ == '__main__':
    # some small testing code
    result = ResultProcessing("../dataset/RuiJin_Processed.csv", "../dataset/lgp_filtered.pkl")
    result.load_models()
    X, y, names = result.readDataRuiJinAD()
    result.calculate_featureList_and_calcvariableList()
    # prog_index, acc_scores =  result.get_accuracy_given_length(1)
    m, i = result.get_feature_co_occurences_matrix(5)
    print(m)
    print(len(i))


