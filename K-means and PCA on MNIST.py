#!/usr/bin/env python
# coding: utf-8

# ## K-Means

# In[97]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd 


# In[12]:


mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0   


# In[13]:


fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(X_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
fig


# In[16]:


X_train.shape


# In[17]:


y_train.shape


# In[18]:


X_test.shape 


# In[19]:


y_test.shape


# In[21]:


X = X_train.reshape(len(X_train),-1)
Y = y_train


# In[22]:


X = X.astype(float) / 255


# In[23]:


X.shape


# In[47]:


from sklearn.cluster import MiniBatchKMeans

n_digits = len(np.unique(y_test))
print(n_digits)

# Initialize KMeans model

kmeans = MiniBatchKMeans(n_clusters = n_digits)

# Fit the model to the training data

kmeans.fit(X)

labels__ = kmeans.labels_
print(labels__)


# In[57]:


def infer_cluster_labels(kmeans, actual_labels):
    #actual_labels = Y
    inferred_labels = {}
  
    for i in range(kmeans.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

        print(labels)
        print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))

    return inferred_labels


# In[58]:


def infer_data_labels(X_labels, cluster_labels):
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)  # empty array of len(X)
    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
            
    return predicted_labels


# In[59]:


cluster_labels = infer_cluster_labels(kmeans, Y)
X_clusters = kmeans.predict(X)
predicted_labels = infer_data_labels(X_clusters, cluster_labels)
print (predicted_labels[:20])
print (Y[:20])


# In[60]:


from sklearn import metrics


# In[61]:


def calculate_metrics(estimator, data, labels):
    print('Number of Clusters: {}'.format(estimator.n_clusters))
    print('Inertia: {}'.format(estimator.inertia_))
    print('Homogeneity: {}'.format(metrics.homogeneity_score(labels, estimator.labels_)))
    


# In[62]:


clusters = [10, 16, 36, 64, 144, 256]


# In[65]:


for n_clusters in clusters:
    estimator = MiniBatchKMeans(n_clusters = n_clusters)
    estimator.fit(X)
    calculate_metrics(estimator, X, Y)
    cluster_labels = infer_cluster_labels(estimator, Y)
    predicted_Y = infer_data_labels(estimator.labels_, cluster_labels)


# In[68]:


X_test_ = X_test.reshape(len(X_test),-1)


# In[72]:


X_test_.shape


# In[73]:


kmeans = MiniBatchKMeans(n_clusters = 256)
kmeans.fit(X)
cluster_labels = infer_cluster_labels(kmeans, Y)


# In[74]:


test_clusters = kmeans.predict(X_test_)
predicted_labels = infer_data_labels(kmeans.predict(X_test_), cluster_labels)


# In[75]:


print('Accuracy: {}\n'.format(metrics.accuracy_score(y_test, predicted_labels)))


# In[76]:


kmeans = MiniBatchKMeans(n_clusters = 36)
kmeans.fit(X)


# In[77]:


centroids = kmeans.cluster_centers_


# In[78]:


images = centroids.reshape(36, 28, 28)
images *= 255
images = images.astype(np.uint8)


# In[79]:


cluster_labels = infer_cluster_labels(kmeans, Y)


# ## PCA

# In[81]:


from sklearn.preprocessing import StandardScaler


# In[84]:


standardized_data = StandardScaler().fit_transform(X)
print(standardized_data.shape)


# In[86]:


cov_matrix = np.matmul(X.T, X) #covariance matrix


# In[87]:


print(cov_matrix.shape)


# In[89]:


from scipy.linalg import eigh


# In[91]:


values, vectors = eigh(cov_matrix, eigvals=(782,783))


# In[92]:


vectors.shape


# In[93]:


vectors = vectors.T


# In[94]:


vectors.shape


# In[96]:


new_coords = np.matmul(vectors, X.T)


# In[98]:


new_coords = np.vstack((new_coords, labels)).T


# In[100]:


dataframe = pd.DataFrame(data=new_coords, columns=("1st_principal", "2nd_principal", "label"))


# In[101]:


print(dataframe.head())


# In[105]:


import seaborn as sn


# In[119]:


sn.FacetGrid(dataframe, hue="label", height=6).map(plt.scatter, "1st_principal" "2nd_principal").add_legend()


# In[110]:


from sklearn import decomposition


# In[115]:


pca = decomposition.PCA() # PCA for dimensionality redcution (non-visualization)
pca.n_components = 784
pca_data = pca.fit_transform(X)
percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
cum_var_explained = np.cumsum(percentage_var_explained)


# In[ ]:




