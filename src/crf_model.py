import pandas as pd
import numpy as np
# Data visualisation
import plotly.express as px
# Modeling
from sklearn_crfsuite import CRF, scorers, metrics
from sklearn.model_selection import cross_val_score,cross_val_predict,RandomizedSearchCV
import scipy.stats
from sklearn.metrics import make_scorer
import pickle
from joblib import dump, load

train_df = pd.read_csv("data/processed/bilou_format/train.tsv", sep="\t",header=0,encoding="utf-8")
dev_df = pd.read_csv("data/processed/bilou_format/dev.tsv", sep="\t",header=0,encoding="utf-8")
test_df = pd.read_csv("data/processed/bilou_format/test.tsv", sep="\t",header=0,encoding="utf-8")

print(f"The training dataset has {len(train_df)} records")
print(f"The dev dataset has {len(dev_df)} records")
print(f"The test dataset has {len(test_df)} records")
train_dev_df = pd.concat([train_df,dev_df],ignore_index=True)

# A class to retrieve the sentences from the dataset
class getsentence(object):
    
    def __init__(self, data):
        self.n_sent = 1.0
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["token"].values.tolist(),
                                                           s["pos_tag"].values.tolist(),
                                                           s["ner_tag"].values.tolist())]
        self.grouped = self.data.groupby("ex#").apply(agg_func)
        self.sentences = [s for s in self.grouped]


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

train_sent = getsentence(train_dev_df).sentences
#dev_sent = getsentence(dev_df).sentences
test_sent = getsentence(test_df).sentences


X_train = [sent2features(s) for s in train_sent]
y_train = [sent2labels(s) for s in train_sent]

X_test = [sent2features(s) for s in test_sent]
y_test = [sent2labels(s) for s in test_sent]

########### Step 1########################
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=False
)
crf.fit(X_train, y_train)

#Calculate averaged F1 scores for all labels except for O
labels = list(crf.classes_)
labels.remove('O')

# save the model to disk
#pickle.dump(crf)

dump(crf, 'nerCRF.joblib')

y_pred = crf.predict(X_test)
metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)

# group B and I results
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

# Load back the pickled model crf = load('nerCRF.joblib')

############ Step 2 #########################
# Hyperparameter optimization

# define fixed parameters and parameters to search
# crf = CRF(
#     algorithm='lbfgs',
#     max_iterations=100,
#     all_possible_transitions=True
# )
# params_space = {
#     'c1': scipy.stats.expon(scale=0.5),
#     'c2': scipy.stats.expon(scale=0.05),
# }

# # use the same metric for evaluation
# f1_scorer = make_scorer(metrics.flat_f1_score,
#                         average='weighted', labels=labels)

# # search
# rs = RandomizedSearchCV(crf, params_space,
#                         cv=3,
#                         verbose=1,
#                         n_jobs=-1,
#                         n_iter=50,
#                         scoring=f1_scorer)
# rs.fit(X_train, y_train)


# # crf = rs.best_estimator_
# print('best params:', rs.best_params_)
# print('best CV score:', rs.best_score_)
# print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

# # Apply the best estimator on the test data
# crf = rs.best_estimator_
# y_pred = crf.predict(X_test)

# # group B and I results
# sorted_labels = sorted(
#     labels,
#     key=lambda name: (name[1:], name[0])
# )
# print(metrics.flat_classification_report(
#     y_test, y_pred, labels=sorted_labels, digits=3
# ))
