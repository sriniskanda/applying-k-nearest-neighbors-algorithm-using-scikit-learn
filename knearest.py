
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# taking data from csv file

#training file
training_file = pd.read_csv("training.csv",names=np.arange(25))
train_data = training_file.iloc[:,:-1]
train_label = training_file[24]
X_train = train_data.values
y_train = train_label.values



#testing file 
test_file = pd.read_csv("test.csv",names=np.arange(24))
X_test = test_file.values
x_testing = X_test[0]

#apllying knn algorithm to train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',metric_params=None, n_neighbors=10, p=2, weights='uniform')

#testing value on test dataset
prediction = knn.predict(X_test)


for i in prediction:
    
    print(i)
