import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

def generate_data():
    data = pd.read_csv('../dataset/datasetCE.csv', encoding="ISO-8859-1" , names=["target", "ids", "date", "flag", "user", "text"])
    df = pd.DataFrame(columns=['Text', 'Label'])
    df['Text']  = data['text']
    df['Label'] = data['target']
    df['Text'] = df['Text'].str.lower().replace('\W',' ', regex=True)
    print(df)

    return df

def SVM(X_train, y_train, X_test, y_test):
    text_clf_svm = Pipeline([('vect', CountVectorizer(max_df=0.85,max_features=1000)), ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=500,tol=1e-3))])

    text_clf_svm = text_clf_svm.fit(X_train, y_train)
    parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}
    gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1,cv=2)
    gs_clf_svm = gs_clf_svm.fit(X_train, y_train)
    print("Melhores parâmetros utilizando SVM com SGD:")
    print(gs_clf_svm.best_params_)
    ypred2 = (gs_clf_svm.best_estimator_).predict(X_test)
    print("\n""Acurácia de classificação média:")
    acc = accuracy_score(y_test, ypred2)
    print(acc)
    print(classification_report(y_test, ypred2))
	#Plota a matriz de confusão
    cm = confusion_matrix(y_test, ypred2)
    cm = sns.heatmap(cm, annot=True)
    #plt.matshow(cm)
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('SVM CONFUSION MATRIX')
    plt.suptitle(acc)
    plt.show()

def main():
    df = generate_data()
    X = df['Text']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    SVM(X_train,y_train,X_test,y_test)

if __name__ == '__main__':
    main()