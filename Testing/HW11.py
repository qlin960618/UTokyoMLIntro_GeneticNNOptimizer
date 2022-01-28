#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os, sys
import matplotlib.pyplot as plt


# In[2]:


import tensorflow as tf


# In[3]:


cifar10_mnist = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10_mnist.load_data()


# In[4]:


x_train, x_test = x_train/255.0, x_test/255.0


# In[5]:


model = tf.keras.models.Sequential([
    tf.keras.Input(shape=x_train.shape[1:]),
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(1,1), activation="relu"),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.summary()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[6]:


history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=60)


# In[7]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.3, 1])
plt.legend()
plt.show()


# In[8]:


model.evaluate(x_test, y_test)


# In[ ]:





# In[ ]:
