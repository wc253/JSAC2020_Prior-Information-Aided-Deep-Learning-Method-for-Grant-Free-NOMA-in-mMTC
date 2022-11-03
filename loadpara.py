# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 09:59:22 2020

@author: lenovo
"""


import numpy as np
import scipy.io 

for i in range(20):
    
    para=np.load('LISTA_bg_giidLISTA T='+str(i+1)+' trainrate=0.01.npz')
    
    namelist=para.files  # name of variables
    
    dic={}
    
    for k,d in para.items():
        dic[k[:-2]]=d
    
    scipy.io.savemat('netpara'+str(i+1)+'.mat',dic)               