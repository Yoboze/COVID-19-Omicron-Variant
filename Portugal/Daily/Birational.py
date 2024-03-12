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

df1 = pd.read_csv('Daily_Omicron_Portugal.csv')


##########################################################################################################
# process data
today = '2022-09-11' # Update this to include more data 
days = pd.date_range(start='2022-4-05',end=today)
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


class PINN_ExpAlpha_Birational:
    # Initialize the class
    def __init__(self, t, I,  layers1, X, lb, ub):
        
        self.lb = lb
        self.ub = ub
        
        self.t = t
        
        self.I = I
        
        self.X = X
        
        #self.b = b
        
        
        
       
        
    
        self.layers1 = layers1
        
        
        self.weights1, self.biases1 = self.initialize_NN(layers1)
       
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.beta = tf.Variable([5500.0], dtype=tf.float32)
        self.beta1 = tf.Variable([2570.0], dtype=tf.float32)
        self.kappa = tf.Variable([1.0], dtype=tf.float32)
        self.kappa1 = tf.Variable([1.0], dtype=tf.float32)
        self.d = tf.Variable([1.0], dtype=tf.float32)
        self.d1 = tf.Variable([1.0], dtype=tf.float32)
        self.c1 = tf.Variable([1.0], dtype=tf.float32)
        self.c = tf.Variable([1.0], dtype=tf.float32)
        self.Nf = tf.Variable([1.0], dtype=tf.float32)
        
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.I_tf = tf.placeholder(tf.float32, shape=[None, self.I.shape[1]])
                       
        
        self.I_predB = self.net_Logistic(self.t_tf)
        self.alpha_predB = self.alphaFunc(self.t_tf)
        
        
        self.l1,self.l2  = self.net_l(self.t_tf)
             
        self.loss = tf.reduce_mean(tf.square(self.I_tf - self.I_predB)) + \
                    tf.reduce_mean(tf.square(self.l1)) + \
                    tf.reduce_mean(tf.square(self.l2)) 
                    
                    
            
        
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
        X = self.X
        kappa = self.kappa
        d = self.d
        t1 = t[0:X]
        alp1 = kappa*d / (1 + d*t1)
        
        #b = self.b
        d1 = self.d1
        c1 = self.c1
        Nf = self.Nf
        kappa1 = self.kappa1
        t2 = t[X-1:]
        b1 = kappa1*d1 / (1 + d1*t2)
        b2 = 1 / (1 + ((1 -(c1/Nf))*(1 + d1*t2)**-kappa1))
        alp2 = tf.multiply(b1,b2)
        
        alp =  tf.concat([alp1,alp2],0)
        
        return alp2
        # bound_b = [tf.constant(0.01, dtype=tf.float32), tf.constant(1.0, dtype=tf.float32)] 
        # return bound_b[0]+(bound_b[1]-bound_b[0])*tf.sigmoid(alp2)

    
    
    
    
    def net_l(self, t):
        X = 60
        c = self.c
        c1 = self.c1
        d = self.d
        d1 = self.d1
        beta = self.beta
        beta1 = self.beta1
        kappa = self.kappa
        kappa1 = self.kappa1
        
        
        alpha = self.alphaFunc(t)
        I = self.net_Logistic(t)
        
        I_t = tf.gradients(I, t)[0]
        
        t1 = t[0:X]
        I1 = I[0:X]
        l1 = I1 - (c/(1+beta*(1 + d*t1)**-kappa))
        
        t2 = t[X-1:]
        I2 = I[X-1:]
        l2 = I2 - (((c/(1+beta*(1 + d*X)**-kappa))-(c1/(1+beta1*(1 + d1*X)**-kappa1))+(c1/(1+beta1*(1 + d1*t2)**-kappa1))))
        
        
        
        
        return l1,l2   
    
        
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
                beta1_value = self.sess.run(self.beta1)
                d_value = self.sess.run(self.d)
                d1_value = self.sess.run(self.d1)
                c1_value = self.sess.run(self.c1)
                c_value = self.sess.run(self.c)
                kappa_value = self.sess.run(self.kappa)
                kappa1_value = self.sess.run(self.kappa1)
                Nf_value = self.sess.run(self.Nf)
                start_time = timeit.default_timer()
        
        
        
    def predict(self, t_star):
        tf_dict = {self.t_tf: t_star}
        
        
        I_star = self.sess.run(self.I_predB, tf_dict)
        
        alpha_star = self.sess.run(self.alpha_predB, tf_dict)
        
        #tt_star = self.sess.run(self.tt_pred, tf_dict)
        
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

model = PINN_ExpAlpha_Birational(t_train, I_train, layers1, 1, lb, ub)
model.train(niter)

# prediction
I_predB,  alpha_predB = model.predict(t_train)

mse_train_loss = model.loss_log
rmse_train_loss = np.sqrt(mse_train_loss)
#print("rmse_train_loss:",*["%.8f"%(x) for x in rmse_train_loss[0:400]])

# flatten array
T0B = t.flatten()
T1B = t_train.flatten()


# re-scale data
I0B = np.min(I) + (np.max(I) - np.min(I))*new_I.flatten()
I1B = np.min(I) + (np.max(I) - np.min(I))*I_predB.flatten()
A1B = alpha_predB.flatten()

# convert float to list
T0B = T0B[0:nd].tolist()
T1B = T1B[0:nd].tolist()
I0B = I0B[0:nd].tolist()
I1B = I1B[0:nd].tolist()
A1B = A1B[0:nd].tolist()

print("daysB:",*["%.8f"%(x) for x in T0B[0:nd]])
print("timeB:",*["%.8f"%(x) for x in T1B[0:nd]])
print("casesB:",*["%.8f"%(x) for x in I0B[0:nd]])
print("infectdB:",*["%.8f"%(x) for x in I1B[0:nd]])
print("alphaB:",*["%.8f"%(x) for x in A1B[0:nd]])

beta_valueB = model.sess.run(model.beta)
beta1_valueB = model.sess.run(model.beta1)
kappa_valueB = model.sess.run(model.kappa)
Nf_valueB = model.sess.run(model.Nf)
d_valueB = model.sess.run(model.d)
d1_valueB = model.sess.run(model.d1)
kappa1_valueB = model.sess.run(model.kappa1)
c1_valueB = model.sess.run(model.c1)
c_valueB = model.sess.run(model.c)

# learned parameters
print("betaB:",*["%.8f"%(x) for x in beta_valueB])
print("beta1B:",*["%.8f"%(x) for x in beta1_valueB])
print("kappaB:",*["%.8f"%(x) for x in kappa_valueB])
print("kappa1B:",*["%.8f"%(x) for x in kappa1_valueB])
print("cB:",*["%.8f"%(x) for x in c_valueB])
print("c1B:",*["%.8f"%(x) for x in c1_valueB])
print("dB:",*["%.8f"%(x) for x in d_valueB])
print("d1B:",*["%.8f"%(x) for x in d1_valueB])
print("NfB:",*["%.8f"%(x) for x in Nf_valueB])

##########################################################################################################

### Error
# Coefficient of determination
corr_matrix = np.corrcoef(I0B, I1B)
corr = corr_matrix[0,1]
R_sq = corr**2
print("R_sqB:", R_sq) 

# MAPE
def mean_absolute_percentage_error(k_true, k_pred):
    k_true, k_pred = np.array(k_true), np.array(k_pred)
    return np.mean(np.abs((k_true - k_pred) / k_true))

mapep =  mean_absolute_percentage_error(I0B, I1B)
print("MAPE_B:", mapep)

#EV
ev = (explained_variance_score(I0B, I1B))
print("EV_B:",ev)

#RMSE
RMSE = np.sqrt(mean_squared_error(I0B, I1B))
print("RMSE_B:", RMSE)

print("mse_train_lossB:",*["%.8f"%(x) for x in mse_train_loss])
print("rmse_train_lossB:",*["%.8f"%(x) for x in rmse_train_loss])
