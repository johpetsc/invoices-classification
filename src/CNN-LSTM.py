import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, Embedding, Dense, Dropout, Activation, LSTM, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, losses, preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow_text as tf_text
from attention import Attention
import seaborn as sns

BATCH_SIZE = 2048
VOCAB_SIZE = 30000

def generate_data():
    data = pd.read_csv('../dataset/datasetCE.csv', encoding="ISO-8859-1" , names=["target", "ids", "date", "flag", "user", "text"])
    df = pd.DataFrame(columns=['Text', 'Label'])
    df['Label'] = data['target'].replace({
        0: 0, 
        4: 1})
    df['Text'] = data['text']
    df['Text'] = df['Text'].str.lower().replace('\W',' ', regex=True)
    train = pd.DataFrame(columns=['Text', 'Label'])
    test = pd.DataFrame(columns=['Text', 'Label'])
    train['Text'], test['Text'], train['Label'], test['Label'] = train_test_split(df['Text'], df['Label'], random_state=42, shuffle=True, test_size=0.6)
    train_dataset = tf.data.Dataset.from_tensor_slices((train['Text'].values, train['Label'].values))
    test_dataset = tf.data.Dataset.from_tensor_slices((test['Text'].values, test['Label'].values))

    return train_dataset, test_dataset

def CNN_model(train_dataset, test_dataset, encoder):
    cnn = pd.DataFrame(columns=['CNN'])
    for x in range(2):
        model = Sequential([
                encoder,
                Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
                Conv1D(32, 3, activation='relu'),
                Dropout(0.5),
                MaxPooling1D(pool_size=2),
                Dropout(0.5),
                Dense(32, activation='relu'),
                Dropout(0.5),
                Attention(name='attention_weight'),
                Dropout(0.5),
                Dense(32, activation='relu'),
                Dropout(0.5),
                Dense(32, activation='relu'),
                Dropout(0.5),
                Dense(2, activation='softmax')])

        model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
        
        history = model.fit(train_dataset, epochs=2,
                        validation_data=test_dataset, 
                        validation_steps=30)
        #model.summary()
        test_loss, test_acc = model.evaluate(test_dataset)
        cnn.loc[x] = [test_acc]

    return cnn['CNN']

def LSTM_model(train_dataset, test_dataset, encoder):
    lstm = pd.DataFrame(columns=['LSTM'])
    for x in range(2):
        model = Sequential([
                encoder,
                Embedding(len(encoder.get_vocabulary()), 32, mask_zero=True),
                Bidirectional(LSTM(32, return_sequences=True)),
                Dropout(0.5),
                Dense(32, activation='relu'),
                Dropout(0.5),
                Attention(name='attention_weight'),
                Dropout(0.5),
                Dense(32, activation='relu'),
                Dropout(0.5),
                Dense(32, activation='relu'),
                Dropout(0.5),
                Dense(2, activation='softmax')])
        
        model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
        history = model.fit(train_dataset, epochs=2,
                        validation_data=test_dataset, 
                        validation_steps=30)
        test_loss, test_acc = model.evaluate(test_dataset)
        lstm.loc[x] = [test_acc]

    return lstm['LSTM']

def main():
    train_dataset, test_dataset = generate_data()
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    encoder = TextVectorization(max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))
    vocab = np.array(encoder.get_vocabulary())
    #print(vocab[:20])
    res = pd.DataFrame(columns=['CNN', 'BiLSTM'])
    res['CNN'] = CNN_model(train_dataset, test_dataset, encoder)
    res['BiLSTM'] = LSTM_model(train_dataset, test_dataset, encoder)

    sns.set_theme(style="whitegrid")
    boxplot = sns.boxplot(data = res)
    plt.show()

if __name__ == '__main__':
    main()