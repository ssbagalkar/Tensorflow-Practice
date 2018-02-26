
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


with tf.name_scope("Inputs"):
    x = tf.placeholder("float",name="X")
    y = tf.placeholder("float",name="Y")


# In[3]:


with tf.name_scope("Weights"):
    w = tf.Variable([1.0, 2.0],name='W')


# In[4]:


with tf.name_scope("Model"):
    y_model = tf.add(tf.multiply(x,w[0]),w[1])


# In[5]:


with tf.name_scope("Error"):
    error = tf.squared_difference(y,y_model)


# In[6]:


with tf.name_scope("Optimizer"):
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)


# In[7]:


init = tf.global_variables_initializer()
errors=[]


# In[8]:


with tf.Session() as sess:
    sess.run(init)
    summary = tf.summary.FileWriter("C:\\Users\\saurabh B\\TensorFlow-practice\\Tensorflow-Practice\\Tensorflow_practice\\Convergence\\Summary_Logs",graph=sess.graph),
    for ii in range(1000):
        x_train = tf.random_normal((1,), mean=5, stddev=2.0)
        y_train = x_train * 2 + 6
        x_val, y_val = sess.run([x_train,y_train])
        _, error_val = sess.run([train_op, error], feed_dict={x: x_val, y: y_val})
        errors.append(error_val)
    w_value = sess.run(w)
    print("Predicted model:{a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))


# In[9]:


plt.plot([np.mean(errors[i-50:i]) for i in range(len(errors))])
plt.show()

