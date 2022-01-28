#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os, sys


# In[2]:


import tensorflow as tf


# In[3]:


fashion_mnist = tf.keras.datasets.fashion_mnist 
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[4]:


x_train, x_test = x_train/255.0, x_test/255.0


# In[8]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='sgd', 
              loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[9]:


model.fit(x_train, y_train, epochs=5)


# In[10]:


model.evaluate(x_test, y_test)


# In[ ]:





# In[ ]:




