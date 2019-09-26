# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:15:38 2019

@author: Victor
"""
#==================INTELLIGENCE VIDEO SURVEILANCE HW 1=========================
#run "tensorboard --logdir=logs" to activate server
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
from contextlib import redirect_stdout
from sklearn.metrics import classification_report
option='1'
epochs=100  #20/100
load=True
#tensorboard=keras.callbacks.TensorBoard(log_dir='logs/model_{}_{}'.format(option,datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")))  
tensorboard=TensorBoard(log_dir='logs/model_{}_{}_{}'.format(option,epochs,datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")))

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape
len(train_labels)
train_labels


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.savefig('Figures/sample image (color spectrum).png')
plt.show()


train_images = train_images / 255.0
test_images = test_images / 255.0


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.savefig('Figures/sample image (grayscale).png')
plt.show()

def plot_label_per_class(data,group):
    data=pd.DataFrame(data)
    data.columns=['label']
    
    f, ax = plt.subplots(1,1, figsize=(12,4))
    g = sns.countplot(data.label, order = data["label"].value_counts().index)
    g.set_title('Number of labels for each class ({}).png'.format(group))

    for p, label in zip(g.patches, data["label"].value_counts().index):
        g.annotate(class_names[label], (p.get_x(), p.get_height()+0.1))
    f.savefig('Figures/Number of labels for each class ({}).png'.format(group))
    plt.show()
    
plot_label_per_class(train_labels,'training')

plot_label_per_class(test_labels,'testing')

#=====================Build the model=========================================
def model_(option='1'):
    if option=='1':
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=(28, 28))) 
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
    elif option=='2':        
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        
    model.compile(optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])
    return model

if load==False:
    model=model_(option)
else:
    model = keras.models.load_model('models/model_{}_{}.h5'.format(option,epochs))


model.summary()  
with open('models/model_{}_structure.txt'.format(option), 'w') as f:
    with redirect_stdout(f):
        model.summary()
#%
if option=='2':
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))
#====================FIT AND SAVE MODELS=======================================
if load==False:
    model.fit(train_images, train_labels, epochs=epochs,validation_split=0.2, callbacks=[tensorboard])
    model.save('models/model_{}_{}.h5'.format(option,epochs))

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)

class_probability = model.predict(test_images)
predictions=np.zeros(len(class_probability))
for i in range(len(class_probability)):
    predictions[i]=np.argmax(class_probability[i,:]).astype(int)
predictions=predictions.astype(int)

correct = np.nonzero(predictions==test_labels)[0]
incorrect = np.nonzero(predictions!=test_labels)[0]
print("Correct predicted classes:",correct.shape[0])
print("Incorrect predicted classes:",incorrect.shape[0])
#%%
train_images = train_images.reshape((60000, 28, 28))
test_images = test_images.reshape((10000, 28, 28))


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                        100*np.max(predictions_array),
                        class_names[true_label],i),
                        color=color)
   

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')



def get_label_confidence(data_index,comment):
    for _, indx in enumerate(data_index[:4]):
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plot_image(indx, class_probability[indx], test_labels, test_images)
        plt.subplot(1,2,2)
        plot_value_array(indx, class_probability[indx],  test_labels)
        plt.savefig('Figures/model_{}_{}_{}_{}.png'.format(option,epochs,indx,comment))
        plt.show()

get_label_confidence(correct,'correctly_predicted')

get_label_confidence(incorrect,'incorrectly_predicted')


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, class_probability[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, class_probability[i], test_labels)
plt.tight_layout()
plt.savefig('Figures/model_{}_{}_multiple_sample.png'.format(option,epochs))
plt.show()


target_names = ["Class {} ({}) :".format(i,class_names[i]) for i in range(10)]
print(classification_report(test_labels, predictions, target_names=target_names))

#====================Error Analysis===========================================
def plot_images(data_index,comment,cmap="Blues"):
    # Plot the sample images now
    f, ax = plt.subplots(4,4, figsize=(16,16))

    for i, indx in enumerate(data_index[:16]):
        ax[i//4, i%4].imshow(test_images[indx].reshape(28,28), cmap=cmap)
        ax[i//4, i%4].axis('off')
        ax[i//4, i%4].set_title("True:{} Pred:{} |no:{}".format(class_names[test_labels[indx]],class_names[predictions[indx]],indx))
    plt.savefig('Figures/model_{}_{}_{}.png'.format(option,epochs,comment))
    plt.show()    
    
plot_images(correct,'correctly_predicted', "Blues")

plot_images(incorrect,'incorrectly_predicted', "Reds")




