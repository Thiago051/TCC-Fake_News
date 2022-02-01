import pandas as pd
from sklearn import metrics
import pickle
from pre_proccess import proccess_text
from confusion_matrix import plot_confusion_matrix

# load classification model
with open('../modelos_treinados/nb.pkl', 'rb') as file:
    vectorizer, classifier = pickle.load(file)

# validation dataset
df = pd.read_csv('corpus_validacao.csv', encoding='UTF-8')

# proccess text
p_text = []
for text in df.text:
    p_text.append(proccess_text(text))

# vectorize text
v_text = vectorizer.transform(p_text)

# classify text
predicted = classifier.predict(v_text)

# calculate metrics
accuracy = metrics.accuracy_score(df.label,predicted)
print(f'Accuracy: {round(accuracy*100,2)}%')
recal = metrics.recall_score(df.label,predicted)
print(f'Recall (sensibilidade): {round(recal*100,2)}%')
specificity = metrics.recall_score(df.label,predicted,pos_label=0)
print(f'Specificity: {round(specificity*100,2)}%')
precision = metrics.precision_score(df.label,predicted)
print(f'Precision: {round(precision*100,2)}%')
f1score = metrics.f1_score(df.label,predicted)
print(f'f1-score: {round(f1score*100,2)}%')

# plot confusion matrix
cm = metrics.confusion_matrix(df.label, predicted, labels=[1, 0])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])