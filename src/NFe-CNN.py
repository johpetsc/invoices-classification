import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from attention import Attention

BATCH_SIZE = 128
VOCAB_SIZE = 10000

def generate_data():
    data = pd.read_excel('../dataset/dataset.xlsx')
    df = pd.DataFrame(columns=['Text', 'Label'])
    # filters the NCM of 9 specific types of products
    data = data[(
         data['prod_ncm'] == 30049099) | # Cremes de beleza, cremes nutritivos e loções tônicas
        (data['prod_ncm'] == 30049069) | # Desodorantes corporais e antiperspirantes, líquidos
        (data['prod_ncm'] == 27101259) | # Outros produtos de beleza ou de maquilagem preparados e preparações para conservação ou cuidados da pele
        (data['prod_ncm'] == 27101921) | # Xampus para o cabelo
        (data['prod_ncm'] == 30049079) | # Produtos de maquilagem para os lábios
        (data['prod_ncm'] == 87089990) | # Águas-de-colônia
        (data['prod_ncm'] == 90211020) | # Sombra, delineador, lápis para sobrancelhas e rímel
        (data['prod_ncm'] == 39174090) | # Sabões de toucador em barras, pedaços ou figuras moldados
        (data['prod_ncm'] == 28044000) ] # Preparações para manicuros e pedicuros
    df['Text']  = data['prod_desc']
    # replaces the NCM code for a label
    df['Label'] = data['prod_ncm'].replace({
        30049099: 0, 
        30049069: 1, 
        27101259: 2, 
        27101921: 3,
        30049079: 4,
        87089990: 5,
        90211020: 6,
        39174090: 7,
        28044000: 8})
    # data cleaning 
    df['Text'] = df['Text'].str.lower().replace('\W',' ', regex=True)
    # splits the data for text and training
    train = pd.DataFrame(columns=['Text', 'Label'])
    test = pd.DataFrame(columns=['Text', 'Label'])
    train['Text'], test['Text'], train['Label'], test['Label'] = train_test_split(df['Text'], df['Label'], random_state=42, shuffle=True, test_size=0.2)
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


def CNN_model(train_dataset, test_dataset, encoder):
    # CNN model with 2 convolutional 1D layers (for text data), an attention layer, and a dense layer with softmax activation for multiclass classification (9 different labels)
    model = Sequential([
            encoder,
            Embedding(len(encoder.get_vocabulary()), 128, mask_zero=True),
            Conv1D(64, 3, activation='relu'),
            Conv1D(32, 3),
            MaxPooling1D(pool_size=2),
            Attention(name='attention_weight'),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(9, activation='softmax')])

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset, 
                    validation_steps=30)
    model.summary()
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
    plt.matshow(cm)
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('CNN CONFUSION MATRIX')
    plt.suptitle(test_acc)
    plt.colorbar()
    plt.show()

def main():
    train_dataset, test_dataset = generate_data()
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    encoder = TextVectorization(max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))
    vocab = np.array(encoder.get_vocabulary())
    #print(vocab[:20])
    
    CNN_model(train_dataset, test_dataset, encoder)

if __name__ == '__main__':
    main()