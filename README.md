# ride-safety-classification
Detects dangerous driving from telematics data using machine learning 

This repository contains submission codes for Grab AI For SEA Challange(Theme: Safety)

## 1. Requirements
2. numpy
3. pandas
4. tensorflow
5. scikit-learn
6. matplotlib
7. scikit-plot

## 2. Descriptions
A LSTM neural network model was developed to perform binary classification on given the telematics data to determine safe or dangerous driving. Deep neural network was selected to avoid the needs for manual hand engineered features.

### A. Data preprocessing
preprocessing.py was used to read the features and labels from the csv files and convert them into numpy array. Irrelevant information is removed from the features. sliding window is used to prepare the input array for time series classification. Hence, the features array is reshaped to (number of samples, length of time steps, number of features). For the given datasets, the input array has the shape of 
(20 000, 119, 9). 

Since the given data has unbalanced number of postive and negative samples (more negative than positive), the negative samples was downsampled. The input array used for the model development has the shape of (9979, 119, 9).<br /> 
[Line 53 - line 56 and line 62 - line 64 need to be uncommented for the downsampling to take effect] 

### B. Model development and training
train.py was used to develop LSTM model for the safe or dangerous driving classification and output a trained keras model in 'model' folder

### C. Evaluation
evaluate.py was used to evaluate the performance of the trained model on the given datasets. The trained model performance is as follows:<br />
Accuracy: 0.6587 <br />
Precision:  0.3960036330608538 <br />
Recall:  0.698578009212898 <br />
F1 score:  0.5054706180711543 <br />


