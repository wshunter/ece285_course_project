import numpy as np


def rotmat(th):
    c=np.cos(th)
    s=np.sin(th)
    return np.asarray([[c,s],[-s,c]])
def delrotmat(th):
    c=np.cos(th)
    s=np.sin(th)
    return np.asarray([[-s,c],[-c,-s]])

def H_between2(xi,xj,meas,CINV=np.identity(3),no_jac = False):
    #loss
    delta = np.zeros((3,1))
    R = rotmat(xi[2]).T
    dR = delrotmat(xi[2]).T
    delta_pos = (xj[:2] - xi[:2])
    delta[:2,0] = (R @ delta_pos) - meas[:2,0]
    delta[2] = (xj[2] - xi[2] - meas[2] + np.pi) %(2*np.pi) - np.pi
    deltc = delta.T @ CINV
    if no_jac:
        return (deltc @ delta)[0,0]
    #loss jacobian
    del_i = np.zeros((1,3))
    del_j = np.zeros((1,3))

    del_H_i = np.zeros((3,3))
    del_H_i[:2,:2] = -R
    del_H_i[2,2] = -1
    del_H_i[:2,2] = dR @ delta_pos

    del_H_j = np.zeros((3,3))
    del_H_j[:2,:2] = R
    del_H_j[2,2] = 1

    return (deltc @ delta)[0,0], 2*deltc @ del_H_i, 2*deltc @ del_H_j

def H_prior2(xi,meas,CINV=np.identity(3),no_jac = False):
    #loss
    delta = np.zeros((3,1))
    delta[:2,0] = xi[:2] - meas[:2,0]
    delta[2] = (xi[2] - meas[2] + np.pi) %(2*np.pi) - np.pi

    deltc = delta.T @ CINV
    if no_jac:
        return (deltc @ delta)[0,0]
    #loss jacobian
    del_H_i = np.identity(3)
    return (deltc @ delta)[0,0], 2*deltc @ del_H_i
