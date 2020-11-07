from lib import build_text_collections, palabras_relevantes
import pandas as pd
import os

data_path="data"

def dataset_tfidf():
    textos=build_text_collections()
    df=palabras_relevantes(textos)
    filename="dataset_tfidf.csv"
    df.to_csv(os.path.join(data_path, filename))


if __name__ == "__main__":
    dataset_tfidf()