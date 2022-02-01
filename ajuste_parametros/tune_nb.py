# grid searching key hyperparametres for SVM
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# define dataset
df = pd.read_csv('../csv_treino/Fake_br-Corpus.csv')
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df.text) #.toarray()
y = df.label

classifier = MultinomialNB()

grid_values = [
                {'alpha':[0,0.00001,0.0001,0.001,0.01,1]}
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
df.to_csv('params_nb.csv', index=True)


# Best Params:  {'alpha': 1}
# Best Estimator:  MultinomialNB(alpha=1)
# Best Score: 0.8568990729808282


# https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/