"""
This module includes functions to read and write localisation tables 

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

import xml.etree.ElementTree as ET

def parse_rapidstorm_xml(filename):
    with open(filename) as f:
        l=f.readline()
    names = []
    root = ET.fromstring(l[2:])
    return root

def get_rapidstorm_attrib(filename,name='identifier'):
    root=parse_rapidstorm_xml(filename)
    all_attribs=[]
    for e in root:
        if name in e.attrib.keys():
            all_attribs.append(e.attrib[name])
        else:
            all_attribs.append(None)
    return all_attribs   
       

def load_rapidstorm_file(filename, usecols=None):
    """
    load locasiation Table and parse header
    *usecols* takes a list of column names to be returned
    """
    names = get_rapidstorm_attrib(filename)
    new_names=[]
    for name in names:
     
        if name == 'Position-0-0':
            name='x'
        elif name == 'Position-1-0':
            name='y'
        elif name == 'Position-2-0':
            name='z'
        else:
            name = name.split('-')[0]
           
        new_names.append(name)
    data=pd.read_csv(filename,names = new_names,delimiter=' ',skiprows=[0])
    return data




def save_rapidstorm_file(filename, usecols=None, precision=None):
    """
    write locasiation Table and write header header
    usecols takes a list of column names to be returned
    precision 
    """
    pass

def to_csv(data,filename):
    data.to_csv(filename, index=False)
