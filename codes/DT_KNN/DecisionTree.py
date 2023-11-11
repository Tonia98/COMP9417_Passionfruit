# individually get train and validation csv file
file_name = 'leaf.csv'
leaf = []
with open(file_name) as file:
  for index,line in enumerate(file):
    if index == 0:
      continue
    if not line.isspace():
      leaf.append(line.strip().split(','))
print(leaf)
target1 = []
feature1 = []
for i in range(len(leaf)):
  target1.append(leaf[i][0])
  feature1.append(leaf[i][1:])
print(target1)
print(feature1)

# individually get train and validation csv file
file_name = 'leaf_test.csv'
leaf = []
with open(file_name) as file:
  for index,line in enumerate(file):
    if index == 0:
      continue
    if not line.isspace():
      leaf.append(line.strip().split(','))
print(leaf)
target2 = []
feature2 = []
for i in range(len(leaf)):
  target2.append(leaf[i][0])
  feature2.append(leaf[i][1:])
print(target2)
print(feature2)

from sklearn.datasets import load_iris
from sklearn import tree
import sys
import os
from IPython.display import Image
import pydotplus
import pandas as pd

# Create a decision tree object,
# using information entropy as a basis
clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=7)
clf = clf.fit(feature1, target1)

feature_name = ['light point','mean_red','mean_green','mean_blue',
          'dark_point','point_size']
target_name = ['Healthy','LeafBlast','Hispa','BrownSpot']

# visualization
dot_data = tree.export_graphviz(clf, feature_names=feature_name,  class_names=target_name, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
print("Accuracy on training set: {:.3f}".format(clf.score(feature1, target1)))
print("Accuracy on training set: {:.3f}".format(clf.score(feature2, target2)))