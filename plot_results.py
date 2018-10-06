import numpy as np
import math
import matplotlib.pyplot as plt 
s_true = np.loadtxt('true_30_10_10_gau.txt')
s_hat = np.loadtxt('shat3.txt')
obs = np.loadtxt('obs_pres.txt')
simul_obs = np.loadtxt('simulobs3.txt')
post_diagv = np.loadtxt('postv.txt')

nx = np.array([30,10,10])

post_std = np.sqrt(post_diagv)

s_true3d = s_true.reshape([nx[2],nx[1],nx[0]])
s_hat3d = s_hat.reshape([nx[2],nx[1],nx[0]])
post_std = post_std.reshape([nx[2],nx[1],nx[0]])

# plz add p_ref so that it looks real log-permeability
for i in range(0,10,2):
    fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True)
    ax[0].pcolor(s_true3d[i,:,:],vmin=-2.0,vmax=1.5, cmap=plt.get_cmap('jet'))
    ax[0].set_title('true lnK in layer %0d' %(i))
    
    ax[1].pcolor(s_hat3d[i,:,:],vmin=-2.0,vmax= 1.5, cmap=plt.get_cmap('jet'))
    ax[1].set_title('estimated ln(pmx)in layer %0d' %(i))
    #fig.savefig('est_lay%0d.png' % (i))
    plt.show()
    plt.close(fig)

i = 4
fig = plt.figure()
plt.pcolor(post_std[i,:,:], cmap=plt.get_cmap('jet'))
plt.title('Uncertainty (posterior std) in lnK estimate, layer %d' % (i))
plt.colorbar()
plt.show()
#fig.savefig('std.png')
plt.close(fig)

# change head?
nobs = obs.shape[0]
fig = plt.figure()
plt.title('obs. vs simul.')
plt.plot(obs, simul_obs, '.')
plt.xlabel('observation')
plt.ylabel('simulation')
minobs = np.vstack((obs, simul_obs)).reshape(-1).min()
maxobs = np.vstack((obs, simul_obs)).reshape(-1).max()
plt.plot(np.linspace(minobs, maxobs, 20), np.linspace(minobs, maxobs, 20), 'k-')
axes = plt.gca()
interval = 1000000
xmin, xmax = math.floor(minobs/interval)*interval, math.ceil(maxobs/interval)*interval
axes.set_xlim([xmin,xmax])
axes.set_ylim([xmin,xmax])
axes.xaxis.set_ticks(np.linspace(xmin,xmax,int(xmax/interval)+1))
axes.yaxis.set_ticks(np.linspace(xmin,xmax,int(xmax/interval)+1))
axes.set_aspect('equal', 'box')
#fig.savefig('obs.png')
plt.show()
plt.close(fig)
