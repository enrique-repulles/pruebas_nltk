import pandas as pd
import datapane as dp

df=pd.read_csv("data/dataset_tfidf.csv",index_col=0)

def top_n_words(df,n):
    important_words=set()
    ''' las N palabras más significativas **de cada categoria**. Luego se juntan todas. '''
    for category in df.columns:
        palabras_categoria=df[category].sort_values(ascending=False).head(n)
        print(palabras_categoria.index)
        important_words.update(palabras_categoria.index)
    return important_words


print(top_n_words(df,30))



dp.Report(
    dp.Markdown("# Ejemplo de la tabla"),
    dp.Table(df.sample()),
    dp.Markdown("# Histogramas de cada categoria"),
    dp.Plot(df.news.hist()),
    dp.Plot(df.learned.hist()),
    dp.Plot(df.fiction.hist())
).save(path='report.html', open=True)
