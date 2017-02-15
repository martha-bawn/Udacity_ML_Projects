#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', '')

### Task 3: Create new feature(s)
for person in data_dict:
    if data_dict[person]['to_messages'] != 'Nan' and data_dict[person]['from_messages'] != 'Nan':
        data_dict[person]['total_messages'] = data_dict[person]['to_messages'] + data_dict[person]['from_messages']
    else:
        data_dict[person]['total_messages'] = 'NaN'
    
    if data_dict[person]['from_poi_to_this_person'] != 'NaN' and data_dict[person]['from_this_person_to_poi'] != 'NaN':
        data_dict[person]['total_poi_messages'] = data_dict[person]['from_this_person_to_poi'] + data_dict[person]['from_poi_to_this_person']
    else:
        data_dict[person]['total_poi_messages'] = 'NaN'
    
    if data_dict[person]['total_messages'] != 'Nan' and data_dict[person]['total_poi_messages'] != 'NaN':
        data_dict[person]['prop_messages_with_poi'] = float(data_dict[person]['total_poi_messages']) / float(data_dict[person]['total_messages'])
    else:
        data_dict[person]['prop_messages_with_poi'] = 'NaN'
### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

### PCA
from sklearn.decomposition import RandomizedPCA
pca = RandomizedPCA(n_components=8, whiten=True).fit(features)
pca.fit(features)
pca_features = pca.transform(features)
# print pca.explained_variance_ratio_

### AdaBoost
from sklearn.ensemble import AdaBoostClassifier
ab_clf = AdaBoostClassifier()
ab_clf.fit(pca_features, labels)
pred = ab_clf.predict(pca_features)
# print "accuracy: ", clf.score(pca_features, labels)

# print "precision: ", precision_score(labels_test, pred)
# print "recall: ", recall_score(labels_test, pred)

# print classification_report(labels_test, pred)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
# Tune AdaBoost
parameters = {'n_estimators': (20, 50, 70, 100)}
sss = StratifiedShuffleSplit()

clf = GridSearchCV(AdaBoostClassifier(), parameters, scoring="f1", cv=sss)
clf.fit(pca_features, labels)

print clf.best_params_
print "F1: ", clf.best_score_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

from tester import test_classifier
test_classifier(clf, my_dataset, features_list)

dump_classifier_and_data(clf, my_dataset, features_list)