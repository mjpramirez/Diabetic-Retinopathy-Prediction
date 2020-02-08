#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install tensorflow')


# In[3]:


get_ipython().system('pip install wrapt --upgrade --ignore-installed')


# In[2]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import keras
from tensorflow.keras.optimizers import RMSprop
from keras.layers import Dropout, Flatten, Dense 
import math 
import numpy as np
from keras import applications 
import random
test_range = random.sample(range(0, 1500), 300)
from keras.layers import Dropout
import datetime
import time
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential 


# In[76]:


# Directory with training pictures
train_dir = '/Users/mariajesusperez/Desktop/fintech_DR/images'
#print(os.getcwd())


# In[170]:


os.chdir('/Users/mariajesusperez/Desktop/fintech_DR')
with open('labels.json', 'r') as file:
    data = json.load(file)
for i, filename in enumerate(os.listdir(train_dir)):
    for j in range(len(data)):
        #print(data[j][0][0:9],filename[0:9],'same?')
        if data[j][0][0:9] == filename[0:9]:
            #os.rename(train_dir +'/'+ filename, train_dir +'/'+str(data[j])+".jpg")
            if data[j][1]==0:
                if i in test_range:
                    os.replace(train_dir +'/'+ filename, '/Users/mariajesusperez/Desktop/fintech_DR/Test_Data/0'+'/'+filename)
                else:
                    os.replace(train_dir +'/'+ filename, '/Users/mariajesusperez/Desktop/fintech_DR/Train_Data/0'+'/'+filename)
            elif data[j][1]==1:
                if i in test_range:
                    os.replace(train_dir +'/'+ filename, '/Users/mariajesusperez/Desktop/fintech_DR/Test_Data/1'+'/'+filename)
                else:
                    os.replace(train_dir +'/'+ filename, '/Users/mariajesusperez/Desktop/fintech_DR/Train_Data/1'+'/'+filename)
            elif data[j][1]==2:
                if i in test_range:
                    os.replace(train_dir +'/'+ filename, '/Users/mariajesusperez/Desktop/fintech_DR/Test_Data/2'+'/'+filename)
                else:
                    os.replace(train_dir +'/'+ filename, '/Users/mariajesusperez/Desktop/fintech_DR/Train_Data/2'+'/'+filename)
            elif data[j][1]==3:
                if i in test_range:
                    os.replace(train_dir +'/'+ filename, '/Users/mariajesusperez/Desktop/fintech_DR/Test_Data/3'+'/'+filename)
                else:
                    os.replace(train_dir +'/'+ filename, '/Users/mariajesusperez/Desktop/fintech_DR/Train_Data/3'+'/'+filename)
            elif data[j][1]==4:
                if i in test_range:
                    os.replace(train_dir +'/'+ filename, '/Users/mariajesusperez/Desktop/fintech_DR/Test_Data/4'+'/'+filename)
                else:
                    os.replace(train_dir +'/'+ filename, '/Users/mariajesusperez/Desktop/fintech_DR/Train_Data/4'+'/'+filename)


# In[2]:


train_dir_real = '/Users/mariajesusperez/Desktop/fintech_DR/Train_Data'
test_dir_real = '/Users/mariajesusperez/Desktop/fintech_DR/Test_Data'

vgg16 = applications.VGG16(include_top=False, weights='imagenet')


# In[3]:


train_datagen = ImageDataGenerator(rescale=1/255)

'''
generator = train_datagen.flow_from_directory(
    train_dir_real,
    target_size=(300, 300), 
    batch_size=128, 
    class_mode=None,
    shuffle=False)

generator_t = train_datagen.flow_from_directory(
    test_dir_real,
    target_size=(300, 300), 
    batch_size=128, 
    class_mode=None,
    shuffle=False)
'''
train_generator = train_datagen.flow_from_directory( 
    train_dir_real, 
    target_size=(300, 300),  # All images will be resized to 150x150
    batch_size=128,
    class_mode='categorical')

test_generator = train_datagen.flow_from_directory( 
    test_dir_real, 
    target_size=(300, 300),  # All images will be resized to 150x150
    batch_size=128,
    class_mode='categorical')

'''
nb_train_samples = len(generator.filenames) 
num_classes = len(generator.class_indices) 
predict_size_train = int(math.ceil(nb_train_samples / 128))  
bottleneck_features_train = vgg16.predict_generator(generator, predict_size_train) 
np.save('bottleneck_features_train.npy', bottleneck_features_train)
nb_train_samples = len(train_generator.filenames)
num_classes = len(train_generator.class_indices)
train_data = np.load('bottleneck_features_train.npy')
train_labels = train_generator.classes
train_labels = to_categorical(train_labels, num_classes=num_classes)

nb_test_samples = len(generator_t.filenames) 
num_classes = len(generator_t.class_indices) 
predict_size_test = int(math.ceil(nb_test_samples / 128))  
bottleneck_features_test = vgg16.predict_generator(generator_t, predict_size_test) 
np.save('bottleneck_features_test.npy', bottleneck_features_test)
nb_test_samples = len(test_generator.filenames)
num_classes = len(test_generator.class_indices)
test_data = np.load('bottleneck_features_test.npy')
test_labels = test_generator.classes
test_labels = to_categorical(test_labels, num_classes=num_classes)
'''


# In[40]:


#Building the model

#model = keras.models.Sequential() 
#model.add(Flatten(input_shape=(300, 300, 3)))
#model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3))) 
#model.add(Dropout(0.5)) 
#model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3))) 
#model.add(Dropout(0.3)) 
#model.add(Dense(5, activation='softmax'))

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    # The sixth convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    # The seventh convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
])
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['acc'])
model.summary()


# In[41]:


#history = model.fit(
#    train_data, 
#    train_labels,
#    epochs=15,
#    batch_size = 128)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1)


# In[42]:


#Plotting Training Data Results
plt.figure()
plt.plot(history.history['acc'])
plt.title('Training Set Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(history.history['loss'])
plt.title('Training Set loss')
plt.ylabel('Loss') 
plt.xlabel('Epoch')

test_loss, test_acc = model.evaluate_generator(test_generator)
print(test_acc)


# In[43]:


test_loss, test_acc = model.evaluate_generator(test_generator)
print(test_acc)


# In[47]:


#Default dimensions we found online
img_width, img_height = 224, 224  
#Create a bottleneck file
top_model_weights_path = 'bottleneck_fc_model.h5'
# loading up our datasets
train_data_dir = '/Users/mariajesusperez/Desktop/fintech_DR/Train_Data'
test_data_dir = '/Users/mariajesusperez/Desktop/fintech_DR/Test_Data'
# number of epochs to train top model 
epochs = 7 #this has been changed after multiple model run 
# batch size used by flow_from_directory and predict_generator 
batch_size = 50 
#Loading vgc16 model
vgg16 = applications.VGG16(include_top=False, weights='imagenet')
#needed to create the bottleneck .npy files
datagen = ImageDataGenerator(rescale=1. / 255) 

start = datetime.datetime.now()
 
generator = datagen.flow_from_directory( 
    train_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode=None, 
    shuffle=False) 
 
nb_train_samples = len(generator.filenames) 
num_classes = len(generator.class_indices) 
 
predict_size_train = int(math.ceil(nb_train_samples / batch_size)) 
 
bottleneck_features_train = vgg16.predict_generator(generator, predict_size_train) 
 
np.save('bottleneck_features_train.npy', bottleneck_features_train)
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)


# In[51]:


#training data
generator_top = datagen.flow_from_directory( 
   train_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=False) 
 
nb_train_samples = len(generator_top.filenames) 
num_classes = len(generator_top.class_indices) 
 
# load the bottleneck features saved earlier 
train_data = np.load('bottleneck_features_train.npy') 
 
# get the class labels for the training data, in the original order 
train_labels = generator_top.classes 
 
# convert the training labels to categorical vectors 
train_labels = to_categorical(train_labels, num_classes=num_classes)


# In[52]:


start = datetime.datetime.now()
 
generator = datagen.flow_from_directory( 
    test_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode=None, 
    shuffle=False) 
 
nb_test_samples = len(generator.filenames) 
num_classes = len(generator.class_indices) 
 
predict_size_test = int(math.ceil(nb_test_samples / batch_size)) 
 
bottleneck_features_test = vgg16.predict_generator(generator, predict_size_test) 
 
np.save('bottleneck_features_test.npy', bottleneck_features_test)
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)


# In[54]:


#test data
generator_top = datagen.flow_from_directory( 
   test_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=False) 
 
nb_train_samples = len(generator_top.filenames) 
num_classes = len(generator_top.class_indices) 
 
# load the bottleneck features saved earlier 
test_data = np.load('bottleneck_features_test.npy') 
 
# get the class labels for the training data, in the original order 
test_labels = generator_top.classes 
 
# convert the training labels to categorical vectors 
test_labels = to_categorical(test_labels, num_classes=num_classes)


# In[159]:


from keras import optimizers


start = datetime.datetime.now()
model = Sequential() 
model.add(Flatten(input_shape=train_data.shape[1:])) 
model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.5)) 
model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.3)) 
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
   optimizer=optimizers.RMSprop(lr=1e-4),
   metrics=['acc'])
history = model.fit(train_data, train_labels, 
    #validation_split=0.2,
   validation_data=(test_data,test_labels),
   epochs=30,
   batch_size=batch_size,
    shuffle=True)
model.save_weights(top_model_weights_path)

end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)


# In[134]:


print(history.history.keys())


# In[160]:


plt.figure()
plt.plot(history.history['acc'],label='Training')

plt.plot(history.history['val_acc'],label='Test')
plt.title('Training & Test Set Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training & Test Set loss')
plt.ylabel('Loss') 
plt.xlabel('Epoch')


# In[161]:


model.summary()


# In[162]:


print(model.evaluate(test_data,test_labels))
print(test_data.size)


# In[153]:


preds = np.round(model.predict(test_data),0)
#preds = model.predict(test_data)
print(preds)


# In[154]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix

labels = ['No DR','Mild NP','Moderate NP','Severe NP','Proliferative']
classification_metrics = metrics.classification_report(test_labels,preds,target_names=labels)
print(classification_metrics)


# In[155]:


import pandas as pd
categorical_test_labels = pd.DataFrame(test_labels).idxmax(axis=1)
categorical_preds = pd.DataFrame(preds).idxmax(axis=1)
df_confusion = pd.crosstab(categorical_test_labels, categorical_preds,rownames=['Actual'], colnames=['Predicted'])
print(df_confusion)


# In[156]:


import matplotlib.pyplot as plt
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_confusion)


# In[4]:


def create_features(img):
    # flatten three channel color image
    color_features = img.flatten()
    # convert image to greyscale
    grey_image = rgb2grey(img)
    # get HOG features from greyscale image
    hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack(color_features)
    return flat_features


# In[5]:


from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2grey


file_path = '/Users/mariajesusperez/Desktop/fintech_DR/Train_Data'
test_dir_real = '/Users/mariajesusperez/Desktop/fintech_DR/Test_Data'
labels=[]
features=[]
count=0
for k in range(0,5):
    for i, filename in enumerate(os.listdir(os.path.join(file_path,str(k)))):
        img_path = os.path.join(file_path,str(k),filename)
        img = Image.open(img_path)
        img = img = img.resize((300,300), Image.ANTIALIAS)
        newi = np.array(img)
        labels.append(k)
        features.append(create_features(newi))
        count+=1
        print(count,k,len(create_features(newi)))
#feature_matrix = np.array(features)
        
        


# In[ ]:





# In[ ]:





# In[197]:


get_ipython().system('pip install -U scikit-image')

import skimage
print(skimage.__version__)


# In[6]:


feature_matrix = np.array(features)
print('Feature matrix shape is: ', feature_matrix.shape)


# In[8]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

start = datetime.datetime.now()
ss = StandardScaler()
feature_matrix = ss.fit_transform(feature_matrix)
#pca = PCA(n_components=500)
#feature_matrix_pca = ss.fit_transform(bees_stand)
print('Feature matrix shape is: ', feature_matrix.shape)
#print('PCA matrix shape is: ', feature_matrix_pca.shape)
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)

#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)


# In[9]:


#HANDING TEST DATA
file_path = '/Users/mariajesusperez/Desktop/fintech_DR/Test_Data'

test_labels=[]
test_features=[]
count=0
for k in range(0,5):
    for i, filename in enumerate(os.listdir(os.path.join(file_path,str(k)))):
        img_path = os.path.join(file_path,str(k),filename)
        img = Image.open(img_path)
        img = img.resize((300,300), Image.ANTIALIAS)
        newi = np.array(img)
        test_labels.append(k)
        test_features.append(create_features(newi))
        count+=1
        print(count,k,len(create_features(newi)))
c


# In[11]:


test_feature_matrix = np.array(test_features)
start = datetime.datetime.now()
ss = StandardScaler()
test_feature_matrix = ss.fit_transform(test_feature_matrix)
#pca = PCA(n_components=300)
#test_feature_matrix_pca = ss.fit_transform(test_bees_stand)
print('Test Feature matrix shape is: ', feature_matrix.shape)
#print('Test PCA matrix shape is: ', feature_matrix_pca.shape)
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)


# In[12]:


start = datetime.datetime.now()
params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
# Performing CV to tune parameters for best SVM fit 
#svm_model = GridSearchCV(SVC(), params_grid, cv=5)
#svm_model.fit(X_train_scaled, Y_train)
#print('Best score for training data:', svm_model.best_score_,"\n") 
#final_model = svm_model.best_estimator_

svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(feature_matrix, labels)
y_pred = svm.predict(test_feature_matrix)
print(y_pred)
#Y_pred = final_model.predict(X_test_scaled)
#Y_pred_label = list(encoder.inverse_transform(Y_pred))
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)


# In[56]:


import pandas as pd
import sklearn
categorical_test_labels = pd.DataFrame(np.array(test_labels))
categorical_preds = pd.DataFrame(y_pred)
df_confusion = sklearn.metrics.confusion_matrix(categorical_test_labels, categorical_preds)
print(df_confusion)
print("\n")
print(sklearn.metrics.classification_report(categorical_test_labels,categorical_preds))
accuracy = sklearn.metrics.accuracy_score(categorical_test_labels, categorical_preds)
print('Model accuracy is: ', accuracy)

print("Training set score for SVM: %f" % svm.score(feature_matrix ,labels))
print("Testing  set score for SVM: %f" % svm.score(test_feature_matrix,test_labels))


# In[64]:


df_confusion = pd.DataFrame(df_confusion)
import matplotlib.pyplot as plt
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

plot_confusion_matrix(df_confusion)


# In[67]:


print(svm)


# In[ ]:





# In[ ]:




