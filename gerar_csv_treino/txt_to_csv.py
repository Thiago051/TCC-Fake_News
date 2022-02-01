import os
import csv
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('punkt')

def proccess_text(text):

    # deixando o texto todo em minúculo
    text = text.lower()

    # tratamento de links
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', text)
    text = re.sub('http','',text)

    # tratamento de emails
    text = re.sub('[\w\+\.\-\~]+@[\w\.\-]+\.\w+', '', text)
    
    # removendo pontuações
    punctuation = string.punctuation+'–‘’“”'
    for p in punctuation:
        text = text.replace(p, '')

    # tratamento de numérais
    text = re.sub(r'[0-9]+\w+[0-9]+\w', '',text) # 17h30m 
    text = re.sub(r'[0-9]+', '', text)
    
    # tokenizando o texto
    word_tokens = word_tokenize(text)
    
    # removendo stop words 
    filtered_sentence = []
    stop_words = set(stopwords.words('portuguese'))
    for word in word_tokens:
        if word not in stop_words:
            filtered_sentence.append(word)
    
    # voltando os dados para forma de texto
    text = " ".join(map(str,filtered_sentence))

    return text
###

# ____ LÊ OS .txt E ESCREVE UM .csv COM OS TEXTOS JÁ PRÉ PROCESSADOS ___ #
def generate_csv(csv_name='new_csv.csv'):

    PATH_FAKE_NEWS = '../Fake.br-Corpus/size_normalized_texts/fake/' # caminho fake news
    PATH_TRUE_NEWS = '../Fake.br-Corpus/size_normalized_texts/true/' # caminho true news
    LABEL_FAKE_NEWS = 1 # classe positiva
    LABEL_TRUE_NEWS = 0 # classe negativa
    QTD_NEWS = 3603

    # cria arquivo .csv
    news_csv = open(f'../csv_treino/{csv_name}.csv','w',newline='',encoding='UTF-8')

    # cria objeto de gravação
    writer = csv.writer(news_csv)

    # grava as linhas [cabeçalho]
    writer.writerow(['index', 'label', 'text'])

    index = 0

    # escrevendo notícias falsas
    for i in range(1,QTD_NEWS): 
        file_name = f'{PATH_FAKE_NEWS}{i}.txt'
        if os.path.isfile(file_name):
            fake_news = open(file_name, 'r', encoding='UTF8')
            news = fake_news.read()
            processed_news = proccess_text(news)
            writer.writerow([index, LABEL_FAKE_NEWS, processed_news])
            index += 1
            fake_news.close()

    # escrevendo noticias verdadeiras
    for i in range(1,QTD_NEWS): 
        file_name = f'{PATH_TRUE_NEWS}{i}.txt'
        if os.path.isfile(file_name):
            true_news = open(file_name, 'r', encoding='UTF8')
            news = true_news.read()
            processed_news = proccess_text(news)
            writer.writerow([index, LABEL_TRUE_NEWS, processed_news])
            index += 1
            true_news.close()

    news_csv.close()

# _________ criando o csv de treino _____#
csv_name = 'Fake_br-Corpus'
generate_csv(csv_name)