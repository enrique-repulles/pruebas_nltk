import pandas as pd
import datapane as dp
import matplotlib

df=pd.read_csv("data/dataset_tfidf.csv",index_col=0)

def top_n_words(df,n):
    important_words=set()
    ''' las N palabras más significativas **de cada categoria**. Luego se juntan todas. '''
    for category in df.columns:
        palabras_categoria=df[category].sort_values(ascending=False).head(n)
        important_words.update(palabras_categoria.index)
    return important_words


print(top_n_words(df,30))

def name(output_filename):
    ''' nombre para un fichero de salida'''
    return "output/analisis_dataset/{}".format(output_filename)


df.describe().to_csv(name("estadisticas.csv"))
df.news.hist()

matplotlib.pyplot.savefig(name("histograma_news.png"))
matplotlib.pyplot.close()

df.learned.hist()
matplotlib.pyplot.savefig(name("histograma_learned.png"))
matplotlib.pyplot.close()
df.fiction.hist()
matplotlib.pyplot.savefig(name("histograma_fiction.png"))
matplotlib.pyplot.close()

