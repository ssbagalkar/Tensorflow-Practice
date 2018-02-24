
# coding: utf-8

# In[7]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


def create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed):
    np.random.seed(seed)
    slices=[]
    centroids=[]
    for i in range(n_clusters):
        samples = tf.random_normal((n_samples_per_cluster, n_features),
                    mean=0.0,stddev=5.0,dtype=tf.float32,seed=seed, name="cluster_{}".format(i))
        current_centroid = (np.random.random((1,n_features)) * embiggen_factor) - (embiggen_factor/2)
        centroids.append(current_centroid)
        samples+=current_centroid
        slices.append(samples)
    # Create a big "samples" dataset
    samples = tf.concat(slices,0,name='samples')
    centroids = tf.concat(centroids,0,name='centroids')
    return centroids, samples
        


# In[9]:


def plot_clusters(all_samples, centroids, n_samples_per_cluster):
    # Plot the different clusters
    color = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
    for i, centroid in enumerate(centroids):
        samples = all_samples[i*n_samples_per_cluster:(i+1)*n_samples_per_cluster]
        plt.scatter(samples[:,0],samples[:,1],c=color[i])
        # Plot centroid
        plt.plot(centroid[0],centroid[1],markersize = 35, marker = 'x', color = 'k', mew = 5)
        plt.plot(centroid[0], centroid[1], markersize = 30, marker = 'x', color = 'm', mew = 2)
    

    


# In[10]:


def choose_random_centroids(samples,n_clusters):
    n_samples = tf.shape(samples)[0]
    random_indices = tf.random_shuffle(tf.range(0,n_samples))
    begin = [0]
    size = [n_clusters]
    centroid_indices = tf.slice(random_indices,begin,size)
    initial_centroids = tf.gather(samples,centroid_indices)
    return initial_centroids
    

