# grid searching key hyperparametres for logistic regression
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

# define dataset
df = pd.read_csv('../csv_treino/Fake_br-Corpus.csv')
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df.text) #.toarray()
y = df.label

classifier = LogisticRegression(random_state=3)

grid_values = [
                {
                'solver':['newton-cg','lbfgs','liblinear','sag','saga'],
                'penalty':['none','l1','l2','elasticnet'],
                'C':[1, 5, 10, 15],
                'l1_ratio': [0,0.5,1]
                }
            ]

grid_classifier = GridSearchCV(
                            estimator=classifier,
                            cv=5,
                            param_grid=grid_values,
                            scoring=['accuracy','precision','recall','f1'],
                            return_train_score=True,
                            refit='f1',
                            verbose=2,
                            n_jobs=-1
                        ) 

grid_classifier.fit(x,y)

print('Best Params: ',grid_classifier.best_params_)
print('Best Estimator: ',grid_classifier.best_estimator_)
print('Best Score:', grid_classifier.best_score_)

df = pd.DataFrame(grid_classifier.cv_results_)[['params',
                                                'mean_test_accuracy',
                                                'mean_test_precision',
                                                'mean_test_recall',
                                                'mean_test_f1']]
df.to_csv('params_reg_log2.csv', index=True)

# ___refit = f1
# Best Params:  {'C': 5, 'l1_ratio': 0, 'penalty': 'l2', 'solver': 'saga'}
# Best Estimator:  LogisticRegression(C=5, l1_ratio=0, random_state=3, solver='saga')
# Best Score: 0.9068006072160593




# https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/