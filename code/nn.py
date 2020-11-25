import csv
import pandas as pd
import numpy as np
import itertools
import plotly
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
#%matplotlib inline



def preprocess(data, test_size):
    x = data.iloc[:,2:32].values
    y = data.iloc[:,1:2].values

    encoder = OneHotEncoder()
    y = encoder.fit_transform(y).toarray()  # Tranforming M/B to binary 1/0
    sc = StandardScaler()                   # Standardise and normalise data
    x = sc.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = test_size) # Split in training and test set
    return x_train, x_test, y_train, y_test

data = pd.read_csv("/Users/anineahlsand/iCloud Drive (arkiv)/Documents/Dokumenter/Documents/Skole/NTNU/Maskinlæring/Project/ML-Project/data/data.csv")
x_train, x_test, y_train, y_test = preprocess(data, 0.1)

def create_model(loss, optimizer, learning_rate):
    model = Sequential()
    model.add(Dense(units=32, input_shape=(30,), activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=4, activation='relu'))
    model.add(Dense(units=2, activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer(lr=learning_rate), metrics=['accuracy'])
    return model

model = create_model('binary_crossentropy', Adam, 0.01)

def train_model(model, x_train, y_train, epochs, batch_size):
    return model.fit(x_train, y_train, shuffle='Logical',epochs=epochs, batch_size=batch_size)

trained_model = train_model(model, x_train, y_train, 100, 32)

def predict(model, x_test):
    return model.predict(x_test)

predicted_y = predict(model, x_test)

#Converting predictions to label
predictions = []
for i in range(len(predicted_y)):
    predictions.append(np.argmax(predicted_y[i]))
#Converting one hot encoded test label to label
test = []
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

def evaluate(y_test, y_pred):
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = evaluate(test,predictions)
print('Accuracy is:', accuracy*100)
print('Recall is:', recall*100)
print('Precision is:', precision*100)
print('F1 is:', f1*100)

cm = confusion_matrix(y_true=test, y_pred=predictions)
def plot_confusion_matrix(cm, classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm_plot_labels = ['Bengin','Malignant']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
