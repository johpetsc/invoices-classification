import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Embedding, Bidirectional
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from attention import Attention
import seaborn as sns

BATCH_SIZE = 1024
VOCAB_SIZE = 30000

def generate_data():
    data = pd.read_csv('../dataset/datasetCE.csv', encoding="ISO-8859-1" , names=["target", "ids", "date", "flag", "user", "text"])
    df = pd.DataFrame(columns=['Text', 'Label'])
    df['Label'] = data['target'].replace({
        0: 0, 
        4: 1})
    df['Text']  = data['text']
    df['Text'] = df['Text'].str.lower().replace('\W',' ', regex=True)
    train = pd.DataFrame(columns=['Text', 'Label'])
    test = pd.DataFrame(columns=['Text', 'Label'])
    train['Text'], test['Text'], train['Label'], test['Label'] = train_test_split(df['Text'], df['Label'], random_state=42, shuffle=True, test_size=0.4)
    train_dataset = tf.data.Dataset.from_tensor_slices((train['Text'].values, train['Label'].values))
    test_dataset = tf.data.Dataset.from_tensor_slices((test['Text'].values, test['Label'].values))
    """for text, label in train_dataset.take(10):
        print ('Text: {}, Label: {}'.format(text, label))
    for text, label in test_dataset.take(10):
        print ('Text: {}, Label: {}'.format(text, label))"""

    return train_dataset, test_dataset

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()


def LSTM_model(train_dataset, test_dataset, encoder):
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
    
    history = model.fit(train_dataset, epochs=5,
                    validation_data=test_dataset, 
                    validation_steps=30)

    test_loss, test_acc = model.evaluate(test_dataset)
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))

    plt.figure(figsize=(16,8))
    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
    y_pred = model.predict(test_dataset)
    predicted_categories = tf.argmax(y_pred, axis=1)

    true_categories = tf.concat([y for x, y in test_dataset], axis=0)

    cm = confusion_matrix(predicted_categories, true_categories)
    cm = sns.heatmap(cm, annot=True)
    #plt.matshow(cm)
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('CNN CONFUSION MATRIX')
    plt.suptitle(test_acc)
    plt.show()

def main():
    train_dataset, test_dataset = generate_data()
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    encoder = TextVectorization(max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))
    vocab = np.array(encoder.get_vocabulary())
    #print(vocab[:20])
    
    LSTM_model(train_dataset, test_dataset, encoder)

if __name__ == '__main__':
    main()