#!/usr/bin/env python3

import numpy as np
#import tensorflow.keras as keras
import pandas as pd
import sys
import tensorflow.compat.v1 as tf
import timeit
import time
import csv
import datetime
import scipy.io
import scipy.optimize
from scipy import optimize
from scipy.interpolate import CubicSpline
from matplotlib.pylab import rcParams
#from statsmodels.tsa.holtwinters import SimpleExpSmothing, Holt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import explained_variance_score
from matplotlib import pyplot as plt

tf.disable_v2_behavior()

##########################################################################################################
# load data

df1 = pd.read_csv('Omicron_Italy.csv')


##########################################################################################################
# process data
today = '2022-05-08' # Update this to include more data 
days = pd.date_range(start='2021-11-30',end=today)
dd = np.arange(len(days))




#total_cases = [df1[day.strftime('%-m/%-d/%Y')][0] for day in days]
total_cases = [df1[day.strftime('%Y-%m-%d')][0] for day in days]


dff =pd.DataFrame(np.array(total_cases))


new_total =dff.rolling(7).mean()
new_total =np.array(new_total[6:]).reshape((-1,1))
firstSix =np.array(total_cases[:6]).reshape((-1,1))
modified_data = np.vstack((firstSix, new_total))    


t = np.reshape(dd, [-1])
I = np.reshape(modified_data, [-1])

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

new_I = NormalizeData(I) # scaled btw 0 and 1

# generating more data points for training
nd = 160
cs1 = CubicSpline(t,new_I)

Td = np.linspace(0,160,nd)

cs_I = cs1(Td)

class PINN_ExpAlpha:
    # Initialize the class
    def __init__(self, t, I,  layers1, lb, ub):
        
        self.lb = lb
        self.ub = ub
        
        self.t = t
        
        self.I = I
        
        self.layers1 = layers1
        
        
        self.weights1, self.biases1 = self.initialize_NN(layers1)
        
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.beta = tf.Variable([7500.0], dtype=tf.float32)
        self.kappa = tf.Variable([1.0], dtype=tf.float32)
        self.d = tf.Variable([1.0], dtype=tf.float32)
        self.Nf = tf.Variable([1.0], dtype=tf.float32)
        
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.I_tf = tf.placeholder(tf.float32, shape=[None, self.I.shape[1]])
        
        
        self.I_predR = self.net_Logistic(self.t_tf)
        self.alpha_predR = self.alphaFunc(self.t_tf)
        
        self.l1 = self.net_l(self.t_tf)
        
        self.loss = tf.reduce_mean(tf.square(self.I_tf - self.I_predR)) + \
                    tf.reduce_mean(tf.square(self.l1)) 
            
            
             
            
        
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.train_op = self.optimizer.minimize(self.loss)
        self.loss_log = []
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, t, layers1, weights1, biases1):
        num_layers = len(layers1)
        
        H = 2.0*(t - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights1[l]
            b = biases1[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights1[-1]
        b = biases1[-1]
        #Y = tf.nn.softplus(tf.add(tf.matmul(H, W), b))
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_Logistic(self, t):
        Logistic = self.neural_net(t, self.layers1, self.weights1, self.biases1)
        I = Logistic
        return I
    
   
    def alphaFunc(self,t):
        d = self.d
        kappa = self.kappa
        b = kappa*d / (1 + d*t)
        return b
    
    
    def net_l(self, t):
        Nf = self.Nf
        beta = self.beta
        kappa = self.kappa
        d = self.d
        alpha = self.alphaFunc(t)
        
        I = self.net_Logistic(t)
        l1 = I - (Nf/(1+beta*(1 + d*t)**-kappa))
        return l1
        
    def train(self, nIter):
        tf_dict = {self.t_tf: self.t, self.I_tf: self.I}
        start_time = timeit.default_timer()

        for it in tqdm(range(nIter)):
            self.sess.run(self.train_op, tf_dict)
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                self.loss_log.append(loss_value)
                beta_value = self.sess.run(self.beta)
                d_value = self.sess.run(self.d)
                kappa_value = self.sess.run(self.kappa)
                Nf_value = self.sess.run(self.Nf)
                start_time = timeit.default_timer()
        
        
        
    def predict(self, t_star):
        tf_dict = {self.t_tf: t_star}
        
        
        I_star = self.sess.run(self.I_predR, tf_dict)
        
        alpha_star = self.sess.run(self.alpha_predR, tf_dict)
        
        return I_star,  alpha_star
        
        

##########################################################################################################
# training the network

niter = 50000  # number of Epochs
layers1 = [1, 64, 64, 64, 64, 1]
t_train = Td.flatten()[:,None]
I_train = cs_I.flatten()[:,None] 
    
#D_train = cs_D.flatten()[:,None]      

# Doman bounds
lb = t_train.min(0)
ub = t_train.max(0)

model = PINN_ExpAlpha(t_train, I_train, layers1, lb, ub)
model.train(niter)

# prediction
I_predR, alpha_predR = model.predict(t_train)


mse_train_loss = model.loss_log
rmse_train_loss = np.sqrt(mse_train_loss)
#print("rmse_train_loss:",*["%.8f"%(x) for x in rmse_train_loss[0:400]])

# flatten array
T0R = t.flatten()
T1R = t_train.flatten()


# re-scale data
I0R = np.min(I) + (np.max(I) - np.min(I))*new_I.flatten()
I1R = np.min(I) + (np.max(I) - np.min(I))*I_predR.flatten()
A1R = alpha_predR.flatten()
#TT1 = np.min(T0) + (np.max(T0) - np.min(T0))*tt

# convert float to list
T0R = T0R[0:nd].tolist()
T1R = T1R[0:nd].tolist()
I0R = I0R[0:nd].tolist()
I1R = I1R[0:nd].tolist()
A1R = A1R[0:nd].tolist()
#TT1 = TT1[0:nd].tolist()

print("daysR:",*["%.8f"%(x) for x in T0R[0:nd]])
print("timeR:",*["%.8f"%(x) for x in T1R[0:nd]])
print("casesR:",*["%.8f"%(x) for x in I0R[0:nd]])
print("infectdR:",*["%.8f"%(x) for x in I1R[0:nd]])
print("alphaR:",*["%.8f"%(x) for x in A1R[0:nd]])

beta_valueR = model.sess.run(model.beta)
kappa_valueR = model.sess.run(model.kappa)
Nf_valueR = model.sess.run(model.Nf)
d_valueR = model.sess.run(model.d)


# learned parameters
print("betaR:",*["%.8f"%(x) for x in beta_valueR])
print("kappaR:",*["%.8f"%(x) for x in kappa_valueR])
print("NfR:",*["%.8f"%(x) for x in Nf_valueR])
print("dR:",*["%.8f"%(x) for x in d_valueR])


##########################################################################################################

##########
### Error
# Coefficient of determination
corr_matrix = np.corrcoef(I0R, I1R)
corr = corr_matrix[0,1]
R_sq = corr**2
print("R_sqR:", R_sq) 

# MAPE
def mean_absolute_percentage_error(k_true, k_pred):
    k_true, k_pred = np.array(k_true), np.array(k_pred)
    return np.mean(np.abs((k_true - k_pred) / k_true))

mapep =  mean_absolute_percentage_error(I0R, I1R)
print("MAPE_R:", mapep)

#EV
ev = (explained_variance_score(I0R, I1R))
print("EV_R:",ev)

#RMSE
RMSE = np.sqrt(mean_squared_error(I0R, I1R))
print("RMSE_R:", RMSE)


print("mse_train_lossR:",*["%.8f"%(x) for x in mse_train_loss])
print("rmse_train_lossR:",*["%.8f"%(x) for x in rmse_train_loss])

