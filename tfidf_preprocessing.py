import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv


def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def apply_tfidf(df, text_column):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])
    
    tfidf_vectors = tfidf_matrix.toarray()
    
    df['tfidf_vector'] = list(tfidf_vectors)
    
    return df

def main():
    csv_file = 'product_reviews_100.csv'
    
    df = load_data(csv_file)
    
    if 'text' not in df.columns:
        print("CSV file must have a 'text' column.")
        return
    
    df = apply_tfidf(df, 'text')
    
    df['tfidf_vector'] =df['tfidf_vector'].apply(lambda arr: list(arr))

    df['tfidf_vector'].to_csv("vector.csv",index=False)

    print("Number of features: ",len(df['tfidf_vector'][1]))

if __name__ == '__main__':
    main()
