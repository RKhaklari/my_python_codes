'''
Your task is to execute the process for proactive detection of fraud while answering following 
questions.
1.   Data cleaning including missing values, outliers and multi-collinearity. 
2.   Describe your fraud detection model in elaboration.
3.   How did you select variables to be included in the model?
4.   Demonstrate the performance of the model by using best set of tools. 
5.   What are the key factors that predict fraudulent customer?
6.   Do these factors make sense? If yes, How? If not, How not? 
7.   What kind of prevention should be adopted while company update its infrastructure?
8.   Assuming these actions have been implemented, how would you determine if they work?

Data dictionary:

*   step - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (31 days simulation).
*   type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER
*   amount - amount of the transaction in local currency
*   nameOrig - customer who started the transaction
*   oldbalanceOrg - initial balance before the transaction
*   newbalanceOrig - new balance after the transaction
*   nameDest - customer who is the recipient of the transaction
*   oldbalanceDest - initial balance recipient before the transaction. Note that there is no information for customers that start with M (Merchants).
*   newbalanceDest - new balance recipient after the transaction. Note that there is no information for customers that start with M (Merchants).
*   isFraud - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.
*   isFlaggedFraud - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200,000 in a single transaction.
'''

# importing required libraries
import pandas as pd
import numpy as np
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import preprocessing
from tensorflow import keras 
from tensorflow.keras import layers

# importing the data
dataset = pd.read_csv('Fraud.csv')

dataset 

# To check datatypes, number of null values and general information about the data.
dataset.info()

# Since all column names are showing zero value, we conclude that there is no missing value in the dataset.

# Let's see how many rows currently are ticked as Fraud.
dataset[dataset['isFraud'] == 1]

# Current status of rows that are flagged as frauds. 
dataset[dataset['isFlaggedFraud'] == 1]

# To check how many rows should be actually flagged as frauds.
dataset1 = dataset[dataset['amount'] > 200000]
dataset1

# Now change the values of 'isFlaggedFraud' column, for rows of TRANSFER transaction type having amount transfer exceeding 200,000/- value, as 1.

# Now we will change the values of isFlaggedFraud column for rows of TRANSFER transaction having amount transfer exceeding 200,000/- value.
dataset2=dataset.copy()
dataset2['isDuplicate'] = dataset2['isFlaggedFraud']
dataset2.loc[(dataset2['amount'] > 200000)==True, 'isDuplicate'] = 1
dataset2['isFlaggedFraud'] = dataset2['isDuplicate'].copy()
dataset2.drop(['isDuplicate'], axis = 1, inplace = True)

# This dataframe is updated dataset with correct rows flagged as frauds.
dataset2

# Verifying the update
dataset2[dataset2['isFlaggedFraud'] == 1]

# Now we will work with the updated data set: dataset2
# Check categories of "type"
dataset2.type.unique()

# Checking the number of rows in TRANSFER type transaction.
dataset2[dataset2['type'] == "TRANSFER"]

# Checking the number of rows in PAYMENT type transaction.
data1 = dataset2[dataset2['type'] == "PAYMENT"] 
data1

data2 = data1[data1['isFraud'] == 1]
data2
# So no row in "PAYMENT" type transaction is fraud i.e. isFraud ==1

# Checking the number of rows in CASH_OUT type transaction.
data3 = dataset2[dataset2['type'] == "CASH_OUT"]

data3[data3['isFraud'] == 1]
# so there is fraud in CASH_OUT type transaction also.

# Checking the number of rows in DEBIT type transaction.
data4 = dataset2[dataset2['type'] == "DEBIT"]

data4[data4['isFraud'] == 1]
# No fraud in DEBIT type transaction.

# Checking the number of rows in CASH_IN type transaction.
data5 = dataset2[dataset2['type'] == "CASH_IN"]

data5[data5['isFraud'] == 1] 
# As expected, there is no frauds in CASH_IN type transactions

''' Now filtering the data set for only CASH_OUT and TRANSFER type transactions, since only these contain positive fraud and flagged Fraud values. 
Make the data set smaller containing only these two categories of "type" transactions. '''

a = ["TRANSFER","CASH_OUT"]
newDataset = dataset2[dataset2.type.isin(a)] 
newDataset

# Making new dataframe "newDataset1" which has changed categorical data into numeric codes for better processing.
label_encoder = preprocessing.LabelEncoder()
newDataset1 = newDataset.copy()
newDataset1['type_1'] = newDataset1['type'].map({"TRANSFER": 1, "CASH_OUT": 0})
newDataset1['nameOrig_1'] = label_encoder.fit_transform(newDataset1['nameOrig'])
newDataset1['nameDest_1'] = label_encoder.fit_transform(newDataset1['nameDest'])
newDataset1

# Now let's check for outliers and multicollinearity
# For outliers:
# Using boxplot
fig = px.box(newDataset1, x='type', y='amount')
fig.update_layout(autosize=False, width=400, height=850)
fig.show()

df = newDataset1.drop(newDataset1[(newDataset1['amount'] > 2200000)].index)
fig = px.box(df, x='type', y='amount')
fig.update_layout(autosize=False, width=500, height=850)
fig.show()

# For multicollinearity:

# Correlation plot to see which columns influence isFraud column.
# column format = ["step", "type_1", "amount", "nameOrig_1", "oldbalanceOrg", "newbalanceOrig", "nameDest_1", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]
df = newDataset1.iloc[:,[0,11,2,12,4,5,13,7,8,9,10]]
plt.figure(figsize=(10, 10))       
dataplot = sbn.heatmap(df.corr(), cmap="YlGnBu", annot=True)

# The heat map shows that none of the other parameters/columns are strongly correlated with isFraud column.

# Now let us build a neural network model
y = df.isFraud.astype(float)                                                  # the dependent/outcome variable
x = df.drop('isFraud', axis=1).astype(float)                                  # the possibly-independent variables, excluding the dependent variable
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)    # training and test sets

# Checking dimensions of the new data sets
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Using logistic regression
clf = LogisticRegression(max_iter=100, random_state = 0).fit(x_train, y_train)
print(clf.predict(x_test))
print(clf.predict_proba(x_test))

clf.score(x_test, y_test)      # output: 99.8% accuracy

# On training set
y_predict = clf.predict(x_train)
roc_auc_score(y_train, y_predict)  #ROC-AUC score is within range of 0.5 to 1.0

# On test set
y_predict2 = clf.predict(x_test)
roc_auc_score(y_test, y_predict2)  #ROC-AUC score is within range of 0.5 to 1.0

# Two layer model using Keras. First layer consist 200 nodes and second layer has only 1 node, with activations ReLu and Sigmoid function because they give better accuracy.
model = keras.Sequential()
model.add(layers.Dense(10, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

opt = keras.optimizers.Adam(learning_rate=0.001)   #optimizer used = Adam
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

A = model.fit(x_train, y_train, validation_split=0.25, epochs=20, batch_size=200)  
score = model.evaluate(x_test, y_test, verbose=0)

score

# Training accuracy vs validation loss plot:
acc = A.history['accuracy']
loss = A.history['loss']
epochs = range(0,20)
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, loss, label='Validation loss')
plt.title('Accuracy and loss')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()

# Plotting model accuracy against number of epochs:
plt.plot(A.history['accuracy'], label='Training accuracy')
plt.plot(A.history['val_accuracy'], label='Validation loss')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Checking ROC-AUC scores of the model for accuracy. A score between 0.5 and 1.0 is considered good.
# Checking on training set
y_pred1 = model.predict(x_train)
roc_auc_score(y_train, y_pred1)      # 0.9 is pretty good score

# Checking on test set
y_pred2 = model.predict(x_test)
roc_auc_score(y_test, y_pred2)       # 0.9 is pretty good score



'''
Inferences:
1. isFraud shoould have 1 value only for type category: TRANSFER and CASH_OUT.
2. isFlaggedFraud can be marked 1 correctly by code snipet no. 1
3. From the correlation mapping we can say that frauds can be studied through oldbalanceOrg, and isFlaggedFraud can be determined through type of transactcion and amount.
4. Through the machine learning model we can accurately determine the "isFraud" column by 0.997 percent accuracy. 
5. Although the logistic regression is also not bad. And it gives result in less time.
6. To test on new data, we can include the individual rows in the test set and predict with the model. The model can also recaliberate on newer real time data, by simply taking the data source to be an updating database. 
7. To check accuracy of the model for future predictions, various accuracy tests, such as being used here, would be required to run in regular interval.
'''
