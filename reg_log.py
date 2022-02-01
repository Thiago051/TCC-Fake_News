# __________ LOGISTIC REGRESSION __________ #

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import pickle
from confusion_matrix import plot_confusion_matrix

# read corpus
df=pd.read_csv('csv_treino/Fake_br-Corpus.csv',encoding='UTF-8')

#DataFlair - Split the dataset
x = df.text
y = df.label
x_train,x_test,y_train,y_test=train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=3)

#DataFlair - Initialize a vectorizer (TfidfVectorizer or CountVectorizer [bag of words])
vectorizer = TfidfVectorizer() #CountVectorizer() #

#DataFlair - Fit and transform train set, transform test set
vec_train = vectorizer.fit_transform(x_train) 
vec_test = vectorizer.transform(x_test)

print('Total de features: ',len(vectorizer.get_feature_names_out()))

#DataFlair - Initialize a LogisticRegression
            # Best:  {'C': 5, 'penalty': 'l2', 'solver': 'saga'}
classifier = LogisticRegression(C=5, solver='saga') 
classifier.fit(vec_train,y_train)

#DataFlair - Predict on the test set and calculate metrics
y_pred = classifier.predict(vec_test)
accuracy = metrics.accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(accuracy*100,2)}%')
recal = metrics.recall_score(y_test,y_pred)
print(f'Recall (Sensibility): {round(recal*100,2)}%')
specificity = metrics.recall_score(y_test,y_pred,pos_label=0)
print(f'Specificity: {round(specificity*100,2)}%')
precision = metrics.precision_score(y_test,y_pred)
print(f'Precision: {round(precision*100,2)}%')
f1score = metrics.f1_score(y_test,y_pred)
print(f'f1-score: {round(f1score*100,2)}%')

# save model
with open('modelos_treinados/reg_log.pkl', 'wb') as file:
    pickle.dump((vectorizer, classifier), file)

# plot confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred, labels=[1, 0])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])