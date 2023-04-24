# import pandas for calc data-frame 
# import pandas as pd
# import sklearn tree for choose the algorithms
# from sklearn.tree import DecisionTreeClassifier
# # read the file data
# door = pd.read_csv('music.csv')

# # separate the data to be input and output data

# # The input data
# drop = door.drop(columns=['genre'])
# # The output data
# link = door['genre']

# # put the model in Variable
# model = DecisionTreeClassifier()
# # train the model on the input and the output data
# model.fit(drop ,link)
# # ask the model for prediction
# results = model.predict([ [21, 1 ] , [21 , 0] ])

# # print the results 
# print(results)
# Testing the data 

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# read the file data
door = pd.read_csv('music.csv')
# separate the data to be input and output data
# The input data
drop = door.drop(columns=['genre'])
# The output data
link = door['genre']

# Testing select 
drop_train ,  drop_test , link_train , link_test = train_test_split(drop , link , test_size=0.1)
# put the model in Variable
model = DecisionTreeClassifier()
# train the model on the input and the output data
model.fit(drop_train , link_train)
# ask the model for prediction
results = model.predict(drop_test)
score = accuracy_score(link_test , results)
# print the results 
print(score)