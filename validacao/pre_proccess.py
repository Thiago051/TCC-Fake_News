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