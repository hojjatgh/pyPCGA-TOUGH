import tough
import numpy as np
from time import time
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import sys
from os import system
import scipy.io as sio
from pyPCGA import PCGA

data_pmx = sio.loadmat('true_pmx.mat')
s_true = data_pmx['s_true']
s_true = s_true-np.min(s_true)
s_true = s_true/2
pts = data_pmx['pts']
#plt.imshow(s_true[:,6,:].T)
#plt.colorbar()
s_true = s_true.reshape(-1,1)
realization = s_true

#'post_cov':"diag",
x_min = [0,0,0]
x_max = [300,100,20]
nx=np.array([30,10,10])
dx = [10.,10.,2.]
prior_std = 1
prior_cov_scale = [100,100,10]
alpha = 1

#params = {'nx':nx,'dx':dx, 'deletedir':False}
nx = np.array([30, 10, 10])
dx = [10., 10., 2.]
    #m = nx[0]*nx[1]*nx[2]

x_obs = [5,7,9,11,13,15,17,19]
y_obs = [1,3,5,7]
z_obs = [3,5,7,9]
params_tough = {'nx':nx,'dx':dx,'dt_measurements': 3600*24*5,'simul_time': 3600*24*20,
          #'obs_type':['Temperature','Gas Pressure'],
          'obs_type':['Temperature'],
          'x_obs':x_obs, 'y_obs':y_obs, 'z_obs':z_obs,
          'deletedir':False}
forward_model = tough.Model(params_tough)

def kernel(R):
        return alpha * np.exp(-R)
params_pcga = {'R':(0.01)**2, 'n_pc':25,
          'maxiter':5, 'restol':.01,
          'matvec':'FFT','xmin':x_min, 'xmax':x_max, 'N':nx,
          'prior_std':prior_std,'prior_cov_scale':prior_cov_scale,
          'kernel':kernel, 'post_cov':"diag",
          'precond':True, 'LM': True, 
          'parallel':True, 'linesearch' : True,'precision':0.0001,
          'forward_model_verbose': True, 'verbose': True,
          'iter_save': True}

par=False
import time
time_beg=time.time()
obs = forward_model.run(s_true,par)
print('simulation time is = %f' %(time.time()-time_beg))
print('The number of observation are = %d' % len(obs))

s_init = np.mean(s_true)*np.ones((len(s_true),1))

prob = PCGA(forward_model.run, s_init = s_init, pts = pts, params = params_pcga, s_true = s_true, obs = obs)

#s_hat, simul_obs, post_diagv, iter_best = prob.Run()
time_beg=time.time()
s_hat, simul_obs,post_diagv, iter_best = prob.Run()
run_time = time.time()-time_beg
print(run_time)


savemat('res_case1.mat',{'s_hat':s_hat, 'simul_obs':simul_obs, 'post_diagv':post_diagv, 'iter_best':iter_best})