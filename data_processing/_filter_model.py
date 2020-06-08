from data_processing._processing_funcs import ResultProcessing
from linear_genetic_programming.lgp_classifier import LGPClassifier
from sklearn.metrics import accuracy_score
import pickle

result = ResultProcessing("../dataset/RuiJin_Processed.csv", "not important")


pickle_file_path = "../dataset/lgp.pkl"
lgp_models = LGPClassifier.load_model(pickle_file_path)
model_list_raw = [i for i in lgp_models]
accuracy = []
model_list = []
sample_list = []  # record overlapping samples
for m in model_list_raw:
    y_pred = m.predict(result.X)
    score = accuracy_score(result.y, y_pred)
    if score > 0.9:  # only filter
        model_list.append(m)
        m.bestProFitness_ = score

with open('../dataset/lgp_filtered.pkl', 'ab') as output:
    pickle.dump(model_list, output, pickle.HIGHEST_PROTOCOL)