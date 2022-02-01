import matplotlib.pyplot as plt
import numpy as np

# valores das métricas [dados de teste]
#metrics_reglog = [92.5, 91.12, 93.88, 93.72, 92.41]
#metrics_svm = [92.29, 91.12, 93.46, 93.32, 92.21]
#metrics_nb = [88.54, 88.07, 89.01, 88.94, 88.5]

# valores das métricas [dados de validacao]
metrics_reglog = [80, 80, 80, 80, 80]
metrics_svm = [80, 73.33, 86.67, 84.62, 78.57]
metrics_nb = [83.33, 86.67, 80, 81.25, 83.87]

# Definindo a largura das barras
barWidth = 0.25

# Aumentando o gráfico
plt.figure(figsize=(10,5))

# Definindo a posição das barras
r1 = np.arange(len(metrics_reglog))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

### Definindo os valores do gráfico, largura e posição das barras

# Criando as barras
plt.bar(r1, metrics_reglog, color='#6A5ACD', width=barWidth, label='Regressão Logística + TF-IDF')
plt.bar(r2, metrics_svm, color='#6495ED', width=barWidth, label='SVM + TF-IDF')
plt.bar(r3, metrics_nb, color='#00BFFD', width=barWidth, label='Naive Bayes + Bag of Words')

# Adicionando legendas as barras
plt.xlabel('\nMétricas')
plt.yticks([10,20,30,40,50,60,70,80,90,100])
plt.xticks([r + barWidth for r in range(len(metrics_reglog))], ['Acurácia', 'Recall', 'Especificidade', 'Precisão', 'F-score'])
plt.ylabel("Valor ( % )")

# Criando a legenda e exibindo o gráfico 
plt.legend()
plt.show()
