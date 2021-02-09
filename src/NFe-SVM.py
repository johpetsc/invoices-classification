import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import re

def generate_data():
    data = pd.read_excel('../database/BaseTrabIA1.xlsx')
    df = pd.DataFrame(columns=['Text', 'Label'])
    data = data[(data['NCM'] == 33049910) | (data['NCM'] == 33072010)]
    df['Text']  = data['DESCRIÇÃO (NFe)']
    df['Label'] = data['NCM']
    df['Text'] = df['Text'].str.lower()
    df['Text'] = df['Text'].replace('\W',' ', regex=True)
    print(df)

    return df

def get_stop_words(stopwords):
    #Pega a lista de stopwords
    with open(stopwords, 'r') as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

def SVM(X_train, y_train, X_test, y_test, stopwords):
	text_clf_svm = Pipeline([('vect', CountVectorizer(max_df=0.85,max_features=1000)), ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=500,tol=1e-3))])

	text_clf_svm = text_clf_svm.fit(X_train, y_train)
	parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}
	gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1,cv=2)
	gs_clf_svm = gs_clf_svm.fit(X_train, y_train)
	print("Melhores parâmetros utilizando SVM com SGD:")
	print(gs_clf_svm.best_params_)
	ypred2 = (gs_clf_svm.best_estimator_).predict(X_test)
	print("\n""Acurácia de classificação média:")
	print(accuracy_score(y_test, ypred2))
	print(classification_report(y_test, ypred2))
	#Plota a matriz de confusão
	cm = confusion_matrix(y_test, ypred2)
	plt.matshow(cm)
	plt.ylabel('Predict')
	plt.xlabel('True')
	plt.title('MATRIZ DE CONFUSAO SVM')
	plt.colorbar()
	plt.show()

def main():
    df = generate_data()
    X = df['Text']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    stopwords = get_stop_words("../stopwords.txt")

    SVM(X_train,y_train,X_test,y_test,stopwords)

if __name__ == '__main__':
    main()