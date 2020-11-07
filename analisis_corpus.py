import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.text import TextCollection
from nltk.text import Text
import re
import pandas as pd



#nltk.download('brown')
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('gutenberg')
#nltk.download('genesis')

#nltk.corpus.brown.categories()
#nltk.corpus.brown.fileids(categories="news")
#nltk.corpus.brown.fileids(categories="government")
#nltk.corpus.brown.fileids(categories="learned")
#nltk.corpus.brown.fileids(categories="fiction")
#nltk.corpus.brown.raw("ck01")



def tokenize(raw_text):
    result=re.sub(pattern="/[a-z+-]*[ .,$*]", repl=" ",string=raw_text)
    result=word_tokenize(result)
    result=[w.lower() for w in result if w not in stopwords.words('english') and len(w)>3]
    return Text(result)



def build_text_collections():
    text_collections={}
    sample_size=4
    for category in ["news","learned", "fiction"]:
        texts=[]
        for fileid in nltk.corpus.brown.fileids(categories=category)[:sample_size]:
            texts.append(tokenize(nltk.corpus.brown.raw(fileid)))
        text_collections[category]=TextCollection(texts)
    text_collections["all"]=TextCollection(text_collections.values())
    return text_collections



#for key, tc in text_collections.items():
#    print(tc.vocab())



def palabras_relevantes(text_collections):
    tf_idf_categorias={
        "news":{},
        "learned":{},
        "fiction":{}
    }
    for categoria in tf_idf_categorias.keys():
        for word in text_collections["all"].vocab(): #TODO: sólo palabras que aparezcan con cierta frecuencia, y que no salgan en todos los documentos                
            tf_idf_categorias[categoria][word]= text_collections["all"].tf_idf(word,text_collections[categoria])

    #Formato pandas
    series=[]
    print("Inicio")
    for categoria in tf_idf_categorias.keys():
        print(categoria)
        series.append(pd.Series(tf_idf_categorias[categoria],name=categoria))
        print("siguiente")
    df=pd.concat(series,axis=1)
    return df



#TextCollection(text_collections.values()).vocab()


# Usar sklearn.feature_extraction.text.TfidfTransformer de scikit 
