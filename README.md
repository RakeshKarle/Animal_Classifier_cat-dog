![IMG_20230419_205905](https://github.com/RakeshKarle/Animal_Classifier_cat-dog/assets/132128728/65237411-98cb-4a27-ac46-4ba0462e4fb2)# Animal_Classifier_cat-dog

The project involves building a CNN using Keras for cat vs. dog image classification.The model consists of convolutional layers, max-pooling, dropout, and dense layers. It is trained using the Adam optimizer and binary cross-entropy loss. ImageDataGenerator is used for data augmentation. The model achieves high accuracy in classifying new images as cats or dogs.

# Detailed description of the project:

1. Objective: The Cat vs. Dog Image Classifier is a machine learning project that uses Convolutional Neural Networks (CNN) to classify images of cats and dogs. The project is implemented in Python using the Keras library with TensorFlow backend.

2. Mounting Google Drive: The project begins by mounting Google Drive to access the image data stored in the 'data' directory.

3. Building the CNN Model: The neural network model is built using the Keras Sequential API. The model architecture consists of several layers:
a. Convolutional Layers: Two convolutional layers with 32 and 64 filters respectively, using ReLU activation and L2 regularization.
b. Max Pooling Layers: Two max-pooling layers with a pool size of (2, 2).
c. Dropout Layers: Two dropout layers with dropout rate 0.4 and 0.3 respectively to reduce overfitting.
d. Flatten Layer: Flattens the output from convolutional layers into a 1D array.
e. Dense Layers: Two dense layers with 128 neurons and 1 neuron (output layer) with ReLU and sigmoid activation functions respectively.

4. Model Compilation: The model is compiled with the 'adam' optimizer and 'binary_crossentropy' loss function for binary classification. The 'accuracy' metric is used for evaluation.

5. Data Preprocessing: The project uses the ImageDataGenerator from Keras to apply data augmentation and normalization. Training and testing data are loaded as batches using flow_from_directory().

6. Training the Model: The model is trained using the fit() function. The training set is used for training, and the test set is used for validation during the training process. The livelossplot library is utilized to visualize the training and validation loss and accuracy during each epoch.

7. Model Evaluation: The trained model is evaluated using the test set. The accuracy and loss metrics are displayed for both training and validation data.

8. Making Predictions: Finally, the trained model is used to make predictions on new images. Two test images (one cat and one dog) are loaded, preprocessed, and passed through the model. The predictions are then displayed, indicating whether the image is classified as a cat or a dog.

