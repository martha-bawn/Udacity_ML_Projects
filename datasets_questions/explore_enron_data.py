#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import numpy as np

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# number of people
print len(enron_data.keys())

# number of features
print len(enron_data["SKILLING JEFFREY K"])

# number of POIs
count = 0
for name in enron_data:
	if enron_data[name]["poi"] == 1:
		count += 1
print count

# What is the total value of the stock belonging to James Prentice?
print enron_data["PRENTICE JAMES"].keys()
print enron_data["PRENTICE JAMES"]["total_stock_value"]

# How many email messages do we have from Wesley Colwell to 
# persons of interest?
print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

# What's the value of stock options exercised by Jeffrey K Skilling?
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

# Of these three individuals (Lay, Skilling and Fastow), 
# who took home the most money? How much?
print "Skilling: ", enron_data["SKILLING JEFFREY K"]["total_payments"]
print "Lay: ", enron_data["LAY KENNETH L"]["total_payments"]
print "Fastow: ", enron_data["FASTOW ANDREW S"]["total_payments"]

# How are missing values denoted in the dataset?
print "NaN" in enron_data["LAY KENNETH L"].values()

# How many folks in this dataset have a quantified salary? 
# What about a known email address?
count = 0
for name in enron_data:
	if enron_data[name]["salary"] != "NaN":
		count += 1
print"salary: ", count

count = 0
for name in enron_data:
	if enron_data[name]["email_address"] != "NaN":
		count += 1
print"email address: ", count

# How many people in the E+F dataset (as it currently exists) have "NaN"
# for their total payments? What percentage of people in the dataset
# as a whole is this?
count = 0
for name in enron_data:
	if enron_data[name]["total_payments"] == "NaN":
		count += 1
print"total_payments NaN: ", count

# How many POIs in the E+F dataset (as it currently exists) have "NaN"
# for their total payments? What percentage of people in the dataset
# as a whole is this?
count = 0
for name in enron_data:
	if enron_data[name]["total_payments"] == "NaN" and \
	enron_data[name]["poi"]:
		count += 1
print"total_payments poi NaN: ", count

