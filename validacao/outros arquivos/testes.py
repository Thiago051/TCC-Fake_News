import pandas as pd
import re
import csv

df = pd.read_csv(
                    'real.csv',
                     encoding='UTF-8'
                )

# cria arquivo .csv
news_csv = open('real.csv','w',newline='',encoding='UTF-8')

# cria objeto de gravação
writer = csv.writer(news_csv)

# grava as linhas [cabeçalho]
writer.writerow(['index', 'label', 'text'])

index = 15
for text in df.text:
    text = re.sub('\n','',text)
    writer.writerow([index, 0, text])
    index += 1

news_csv.close()
