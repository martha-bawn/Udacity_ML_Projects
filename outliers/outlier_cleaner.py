#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    errors = predictions - net_worths
    max_keep = np.percentile(errors, 90)

    for i in range(len(predictions)):
        if errors[i] <= max_keep:
            cleaned_data.append((ages[i], net_worths[i], errors[i]))
    
    return cleaned_data

