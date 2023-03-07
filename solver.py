import numpy as np
import matplotlib.pyplot as plt
import tqdm
from geom import rotmat, delrotmat, H_between2, H_prior2


N = 31
T = np.arange(N)
pos = np.vstack((np.cos(T*2*np.pi/30 - np.pi/2),np.sin(T*2*np.pi/30 - np.pi/2),T*2*np.pi/30))

#bot was always moving forward and rotating a little bit.
meas = {}
#add relative odom constraints
for i in range(N-1):
    th = (pos[2,i+1] - pos[2,i]    +np.pi)%(2*np.pi)-np.pi
    rel_trans = rotmat(pos[2,i]).T @ (pos[:2,i+1] - pos[:2,i])
    meas[('b',i,i+1)] = np.asarray([[rel_trans[0],rel_trans[1],th]]).T
meas[('b',0,N-1)] = np.zeros((3,1))
meas[('p',0,)] = np.zeros((3,1))
rpos = np.vstack((pos[0,:],pos[1,:],np.cos(pos[2,:]),np.sin(pos[2,:])))

#direct gradient descent solution
pn = np.zeros((3,N))

#measurement loss
COV = np.diag([0.1,0.1,1.0])
CINV = np.linalg.inv(COV)



def H_all(state, meas, no_jac=False):
    loss = 0.0
    grad_H  = np.zeros_like(state)
    for m in meas:
        if m[0] == 'b':
            #pose-pose measurement
            if no_jac:
                loss += H_between2(state[:,m[1]],state[:,m[2]],meas[m],CINV=CINV,no_jac=no_jac)
            else:
                l, jac_i, jac_j = H_between2(state[:,m[1]],state[:,m[2]],meas[m],CINV=CINV,no_jac=no_jac)
                grad_H[:,m[1]] += jac_i[0]
                grad_H[:,m[2]] += jac_j[0]
                loss += l
        if m[0] == 'p':
            #prior
            if no_jac:
                loss += H_prior2(state[:,m[1]], meas[m], CINV=1e1*CINV,no_jac=no_jac)
            else:
                l, jac_i = H_prior2(state[:,m[1]], meas[m], CINV=1e1*CINV,no_jac=no_jac)
                grad_H[:,m[1]] += jac_i[0]
                loss += l
    if no_jac:
        return loss
    else:
        return loss, grad_H



#x0 = np.zeros((3,N))
x0 = np.copy(pos) + np.random.normal(scale=1e-1, size=pos.shape)
ybound = np.inf
xsh = x0.shape
lam = 1.0
losses = []
lams = []

states = [x0]
for i in tqdm.trange(1000):
    loss, grad = H_all(x0, meas)
    losses.append(loss)
    update_state = True
    while(update_state):
        if lam < 1e-20:
            break
        upd = x0 - (np.linalg.pinv(grad.reshape((1,-1))) * lam*loss).reshape(xsh)
        ybound = H_all(upd, meas, no_jac=True)
        #print(f"Tried lam {lam}, {ybound} vs {loss}")
        update_state = False
        if ybound >= loss:
            lam *= 1e-1
        else:
            update_state = False
            lams.append(lam)
    lam *= 2
    #print("succeeded")
    
    x0 = upd
    states.append(np.copy(x0))
        
plt.subplot(1,2,1)
plt.plot(losses)
plt.yscale('log')
plt.subplot(1,2,2)
plt.plot(lams)
plt.yscale('log')
plt.figure()
plt.plot(pos[0,:],pos[1,:], 'k')
plt.plot(x0[0,:],x0[1,:],'b')
plt.gca().set_aspect(1.0)


fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect(1.0)
def animate(idx):
    ax.clear()
    ax.plot(pos[0,:],pos[1,:], 'k')
    ax.plot(states[idx][0,:],states[idx][1,:], 'b')
    ax.set_title(idx)
import matplotlib.animation as ani
anim = ani.FuncAnimation(fig, animate, frames=range(len(states)))
plt.show()
#plt.plot(x,y)
#plt.show()
