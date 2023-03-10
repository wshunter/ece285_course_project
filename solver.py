import numpy as np
import matplotlib.pyplot as plt
import tqdm
from geom import rotmat, delrotmat, H_between2, H_prior2, rotmatC, H_graph, H_graph_conv, solve_graph_convex, solve_graph
from convex import hull_so2, graph_hull_so2#, H_between2_conv

N = 101
T = np.arange(N)
pos = np.vstack((np.cos(T*2*np.pi/(N-1) - np.pi/2),np.sin(T*2*np.pi/(N-1) - np.pi/2),T*2*np.pi/(N-1)))

#meas_noise_xy = np.random.normal
#bot was always moving forward and rotating a little bit.
meas = {}
#add relative odom constraints (noiseless for now)
for i in range(N-1):
    th = (pos[2,i+1] - pos[2,i]    +np.pi)%(2*np.pi)-np.pi
    rel_trans = rotmat(pos[2,i]).T @ (pos[:2,i+1] - pos[:2,i])
    meas[('b',i,i+1)] = np.asarray([[rel_trans[0],rel_trans[1],th]]).T# + np.random.normal(scale=0.01,size=((3,1)))
meas[('b',0,N-1)] = np.zeros((3,1))
meas[('p',0,)] = pos[:,0,np.newaxis]
rpos = np.vstack((pos[0,:],pos[1,:],np.cos(pos[2,:]),np.sin(pos[2,:])))

xstart = 

posC = hull_so2(pos)
measC = graph_hull_so2(meas)

#direct gradient descent solution
pn = np.zeros((3,N))

#measurement loss
COV = np.diag([0.1,0.1,1.0])
CINV = np.linalg.inv(COV)



#x0 = np.zeros_like(posC)
x0[:,0] = posC[:,0]

x0, losses, lams, states = solve_graph_convex(x0,measC)

x0g, lossesg, lamsg, statesg = solve_graph(x0

final_angles = np.arctan2(x0[3,:],x0[2,:])

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
plt.figure()
plt.plot(pos[2,:],color='k')
plt.plot(final_angles, color='b')


fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect(1.0)
def animate(idx):
    ax.clear()
    ax.plot(pos[0,:],pos[1,:], 'k')
    ax.plot(states[idx][0,:],states[idx][1,:], 'b')
    ax.set_title(idx)
import matplotlib.animation as ani
anim = ani.FuncAnimation(fig, animate, frames=range(0,len(states),1))
plt.show()


plt.show()



# state0 = np.asarray([0.0,0.0,1.0,0.0])
# state1 = np.asarray([1.0,0.0,1.0,0.0])
# meas = np.asarray([1.0,0.0,1.0,1.0])[:,np.newaxis]
# dist = 2.0
# npt = 200
# c_space = np.asarray(np.meshgrid(np.linspace(-dist,dist,npt),np.linspace(-dist,dist,npt)))

# losses = np.zeros((npt,npt))

# for si in range(npt):
#     for ci in range(npt):
#         losses[si,ci] = H_between2_conv(state0,np.asarray([c_space[1,si,ci],c_space[0,si,ci],1.0,0.6]),meas,1.0,1.0)

# fig = plt.figure()
# ax = fig.add_subplot()
# ax.imshow(losses.T, origin='lower', extent = [-dist,dist,-dist,dist])
# plt.show()

''' LMQ solver
#all-zero initialization fails to find the global minimum.
x0 = np.zeros((3,N))

#set the noise to a large value (e.g. 1e1) and it will typically converge to a local minimum.

x0 = np.copy(pos) + np.random.normal(scale=1e0, size=pos.shape)
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
        #I am taking the lambda term outside of the pinv as opposed to normal LMQ (so we decrease lambda on a failure) since I find it more natural to formulate the objective function this way. 
        upd = x0 - (np.linalg.pinv(grad.reshape((1,-1))) * lam*loss).reshape(xsh)
        ybound = H_all(upd, meas, no_jac=True)

        #print(f"Tried lam {lam}, {ybound} vs {loss}")
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
anim = ani.FuncAnimation(fig, animate, frames=range(0,len(states),10))
plt.show()
#plt.plot(x,y)
#plt.show()
'''

''' #Scratch code for derivative test


npt = 500
lam = np.linspace(-np.pi,np.pi,npt)
linesp = np.linspace(-1,1,100)
fvals = np.zeros_like(lam)

mth = np.random.uniform()*2*np.pi

X0 = np.random.normal(size=(4,31))

meas = np.asarray([np.random.normal(),np.random.normal(),np.cos(mth),np.sin(mth)])[:,np.newaxis]
for i in range(npt):
    X0[3,1] = lam[i]
    fvals[i] = HC_all(X0,measC)[0]

pnt = -2.0

X0[3,1] = pnt
y, der = HC_all(X0,measC)

line = linesp*der[3,1] + y

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(lam,fvals)
ax.plot(linesp+pnt,line)
#print(HC_all(posC,measC))

plt.show()
#print(HC_all(np.zeros((4,posC.shape[1])),measC))


'''
