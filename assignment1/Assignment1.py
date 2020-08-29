'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float/double/large


def calculate_entropy(p, n):
    entropy = 0
    if p > 0 and p+n > 0:
        entropy += (-p/(p+n))*np.log2(p/(p+n))
    if n > 0 and p+n > 0:
        entropy += (-n/(p+n))*np.log2(n/(p+n))
    return entropy


def get_entropy_of_dataset(df):
    entropy = 0
    p = n = 0
    for i in df.iloc[:, -1]:
        if i == "yes":
            p += 1
        elif i == "no":
            n += 1
    entropy = calculate_entropy(p, n)
    return entropy


'''Return entropy of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float/double/large


def get_entropy_of_attribute(df, attribute):
    entropy_of_attribute = 0
    unique_values = df[attribute].unique()
    for i in unique_values:
        sub_df = df[df[attribute] == i]
        entropy = get_entropy_of_dataset(sub_df)
        entropy_of_attribute += (sub_df.shape[0]/df.shape[0]) * entropy

    return abs(entropy_of_attribute)


'''Return Information Gain of the attribute provided as parameter'''
# input:int/float/double/large,int/float/double/large
# output:int/float/double/large


def get_information_gain(df, attribute):
    information_gain = get_entropy_of_dataset(
        df) - get_entropy_of_attribute(df, attribute)
    return information_gain


''' Returns Attribute with highest info gain'''
# input: pandas_dataframe
# output: ({dict},'str')


def get_selected_attribute(df):

    information_gains = {}
    selected_column = df.columns[0]
    largest_ig = 0

    for i in df.iloc[:, 0:len(df.columns)-1].columns:
        information_gains[i] = get_information_gain(df, i)
        if(largest_ig < information_gains[i]):
            largest_ig = information_gains[i]
            selected_column = i

    '''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''

    return (information_gains, selected_column)


'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''
