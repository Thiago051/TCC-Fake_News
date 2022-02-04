
### Trabalho apresentado ao Curso de Ciência da Computaçãodo Instituto Federal de Educação, Ciência e Tecnologia deBrasília, campus Taguatinga, como requisito parcial paraobtenção do grau de  Bacharel em Ciência da Computação.

# Autores:
* [Thiago Oliveira](https://github.com/Thiago051)
* [Victor Marques](https://github.com/victor35)

# Título do Trabalho: 
* Estudo de Técnicas de Aprendizagem de Máquina para a Detecção de Notícias Falsas em Língua Portuguesa.

# Corpus Utilizados:
* [Fake.Br Corpus](https://github.com/roneysco/Fake.br-Corpus)
* [Fakepedia Corpus](https://github.com/andersoncordeiro/Fakepedia-Corpus)

# Algoritmos de Classificação Testados:
* Regressão Logística
* *SVM*
* *Naive Bayes*

# Modelos de Extração de Caractrísticas de Texto Utilizados:
* *Bag of Words*
* *TF-IDF*

# Melhores Resultados Obtidos:

| Algoritmo de Classificação + Modelo de Extração de Características | Acurácia | Sensibilidade (*Recall*) | Especificidade | Precisão | *F-score* |
|:------------------------------------------------------------------:|:--------:|:------------------------:|:--------------:|:--------:|:---------:|
|                   Regressão Logística + *TF-IDF*                   |  92,50%  |          91,12%          |     93,88%     |  93,72%  |   92,41%  |
|                          *SVM* + *TF-IDF*                          |  92,29%  |          91,12%          |     93,46%     |  93,32%  |   92,21%  |
|                   *Naive Bayes* + *Bag of Words*                   |  88,54%  |          88,07%          |     89,01%     |  88,94%  |   88,50%  |

# Principais Bibliotecas Utilizadas:
* *sklearn*
* *nltk*
* *pandas*
* *numpy*
* *pickle*
* *csv*
* *re*
* *string*
* *os*
