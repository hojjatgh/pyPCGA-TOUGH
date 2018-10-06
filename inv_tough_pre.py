import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tough
from pyPCGA import PCGA
import math

# forward model information: model domain, discretization, observation location and interval
nx = [30, 10, 10]
dx = [10., 10., 2.]

# monitoring indices 
xlocs = np.arange(5,21,2)
ylocs = np.array([1,3,5,7])
zlocs = np.array([3,5,7,9])

forward_model_params = {'nx':nx,'dx':dx, 'deletedir':False, \
'xlocs': xlocs, 'ylocs':ylocs, 'zlocs':zlocs, \
'obs_type':['Gas Pressure'],'t_obs_interval':86400.*5.}


m = nx[0]*nx[1]*nx[2]
N = np.array([nx[0],nx[1],nx[2]])
xmin = np.array([0.,0.,0.])
xmax = np.array([300.,100.,20.])
pts = None # for regular grids, you don't need to specify pts. 

# covairance kernel and scale parameters
prior_std = 0.7
prior_cov_scale = np.array([100.0,100.0,10.0])

def kernel(r): return (prior_std ** 2) * np.exp(-r**2)

# forward model wrapper for pyPCGA
s_true = np.loadtxt("true_30_10_10_gau.txt")
obs = np.loadtxt('obs_pres.txt')

# prepare interface to run as a function
def forward_model(s, parallelization, ncores=None):
    model = tough.Model(forward_model_params)

    if parallelization:
        simul_obs = model.run(s, parallelization, ncores)
    else:
        simul_obs = model.run(s, parallelization)
    return simul_obs

params = {'R': (1000.0) ** 2, 'n_pc': 30,
        'maxiter': 10, 'restol': 0.1,
        'matvec': 'FFT', 'xmin': xmin, 'xmax': xmax, 'N': N,
        'prior_std': prior_std, 'prior_cov_scale': prior_cov_scale,
        'kernel': kernel, 'post_cov': "diag",
        'precond': True, 'LM': True,
        'parallel': True, 'linesearch': True, 'precision': 2e-3,
        'forward_model_verbose': True, 'verbose': True,
        'iter_save': True, 'ncores':6}

# params['objeval'] = False, if true, it will compute accurate objective function
# params['ncores'] = 36, with parallell True, it will determine maximum physcial core unless specified

s_init = np.mean(s_true) * np.ones((m, 1))
# s_init = np.copy(s_true) # you can try with s_true!

# initialize

prob = PCGA(forward_model, s_init, pts, params, s_true, obs)
# prob = PCGA(forward_model, s_init, pts, params, s_true, obs, X = X) #if you want to add your own drift X

# run inversion
s_hat, simul_obs, post_diagv, iter_best = prob.Run()

post_std = np.sqrt(post_diagv)

s_true3d = s_true.reshape([nx[2],nx[1],nx[0]])
s_hat3d = s_hat.reshape([nx[2],nx[1],nx[0]])
post_std = post_std.reshape([nx[2],nx[1],nx[0]])

for i in range(0,10,2):
    fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True)
    ax[0].pcolor(s_true3d[i,:,:],vmin=-2.0,vmax=1.5, cmap=plt.get_cmap('jet'))
    ax[0].set_title('true lnK')
    
    ax[1].pcolor(s_hat3d[i,:,:],vmin=-2.0,vmax=1.5, cmap=plt.get_cmap('jet'))
    ax[1].set_title('estimated ln(pmx)')
    fig.savefig('est_lay%0d.png' % (i))
    plt.close(fig)

i = 4
fig = plt.figure()
plt.pcolor(post_std[i,:,:], cmap=plt.get_cmap('jet'))
plt.title('Uncertainty (posterior std) in lnK estimate, layer %d' % (i))
plt.colorbar()
fig.savefig('std.png')
plt.close(fig)

nobs = prob.obs.shape[0]
fig = plt.figure()
plt.title('obs. vs simul.')
plt.plot(prob.obs, simul_obs, '.')
plt.xlabel('observation')
plt.ylabel('simulation')
minobs = np.vstack((prob.obs, simul_obs)).min(0)
maxobs = np.vstack((prob.obs, simul_obs)).max(0)
plt.plot(np.linspace(minobs, maxobs, 20), np.linspace(minobs, maxobs, 20), 'k-')
plt.axis('equal')
axes = plt.gca()
axes.set_xlim([math.floor(minobs), math.ceil(maxobs)])
axes.set_ylim([math.floor(minobs), math.ceil(maxobs)])
fig.savefig('obs.png')
plt.close(fig)

fig = plt.figure()
plt.semilogy(np.linspace(1,len(prob.objvals),len(prob.objvals)), prob.objvals, 'r-')
plt.xticks(np.linspace(1,len(prob.objvals),len(prob.objvals)))
plt.title('obj values over iterations')
plt.axis('tight')
fig.savefig('obj.png')
plt.close(fig)

