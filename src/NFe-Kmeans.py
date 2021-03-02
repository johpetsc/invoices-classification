import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans

def generate_data():
    data = pd.read_csv('../dataset/NF_Detalhe_PB.csv', sep=";")
    df = pd.DataFrame(columns=['Text'])
    df['Text']  = data['prod_desc']
    labels = data['prod_ncm'].tolist()
    df['Text'] = df['Text'].str.lower()
    df['Text'] = df['Text'].replace('\W',' ', regex=True)
    print(df['Text'].nunique())

    return df, labels

def K_means(X, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    kmeans = KMeans(n_clusters=360, random_state=0)
    kmeans.fit(X)
    pred_classes = kmeans.predict(X)
    res = pd.DataFrame(columns=['cluster', 'label'])
    res['cluster'] = pred_classes
    res['label'] = labels
    res.sort_values(by=['cluster'])
    #pd.set_option('display.max_rows', None)
    print(res.sort_values(by=['cluster']))
    

def main():
    df, labels = generate_data()
    X = df['Text']

    K_means(X, labels)

if __name__ == '__main__':
    main()