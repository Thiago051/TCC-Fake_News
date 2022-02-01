# grid searching key hyperparametres for SVM
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# define dataset
df = pd.read_csv('../csv_treino/Fake_br-Corpus.csv')
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df.text) #.toarray()
y = df.label

classifier = LinearSVC(random_state=3)

grid_values = [
                {
                'penalty':['l1', 'l2'],
                'loss':['hinge','squared_hinge'],
                'C':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15]
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
df.to_csv('params_svm.csv', index=True)


# Best Params:  {'C': 1, 'loss': 'squared_hinge', 'penalty': 'l2'}
# Best Estimator:  LinearSVC(C=1, random_state=3)
# Best Score: 0.9058434920213987