# Electronic Invoices Classification
Compares the performance of two neural network (NN) models, CNN and BiLSTM, when an Attention layer is added. The classification is based on a short-text description of products contained in Brazilian electronic invoices (NF-es). The data is based on texts from 9 different types of products, making this a multi-class problem. A SVM model was also used as means of comparing the values produced from NNs with a more traditional approach.

When evaluating the results, both NNs produced better results than the SVM model, as expected. Furthermore, when comparing the results of both NNs, the CNN model was able to perform faster, better, and more consistently (see results) than the BiLSTM model.

# Requirements
- tensorflow 2.x
- numpy
- pandas
- scikit-learn
- matplotlib