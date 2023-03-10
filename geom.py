import numpy as np
import tqdm

def rotmat(th):
    c=np.cos(th)
    s=np.sin(th)
    return np.asarray([[c,s],[-s,c]])
def delrotmat(th):
    c=np.cos(th)
    s=np.sin(th)
    return np.asarray([[-s,c],[-c,-s]])

def rotmatC(cs):
    return np.asarray([[cs[0],cs[1]],[-cs[1],cs[0]]])

def delrotmatC(cs):
    return np.asarray([[-cs[1],cs[0]],[-cs[0],-cs[1]]])
    
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

# full jacobian of measurement state
def H_graph(state, meas, no_jac=False):
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
                loss += H_prior2(state[:,m[1]], meas[m], CINV=CINV,no_jac=no_jac)
            else:
                l, jac_i = H_prior2(state[:,m[1]], meas[m], CINV=CINV,no_jac=no_jac)
                grad_H[:,m[1]] += jac_i[0]
                loss += l
    if no_jac:
        return loss
    else:
        return loss, grad_H



SO2_COS_INNER_DERIVATIVE = np.identity(2)
SO2_SIN_INNER_DERIVATIVE = np.asarray([[0,1],[-1,0]])

def H_between2_conv(xi,xj,meas,alpha=1.0,beta=1.0):
    RJ = rotmatC(xj[2:])
    RI = rotmatC(xi[2:])
    RIJ = rotmatC(meas[2:,0])
    DELTA_R = RJ - RIJ @ RI
    alpha_term = alpha * np.sum(np.square(DELTA_R))
    
    del_pos = xj[:2] - xi[:2] - RI @ meas[:2,0]
    beta_term = beta * np.sum(np.square(del_pos))
    loss = alpha_term + beta_term

    del_Hc_j = np.zeros((1,4))
    #derivative wrt xj, yj
    del_Hc_j[0,:2] = 2*beta*del_pos

    del_Rj_inner = 2*alpha*(RJ - RIJ @ RI)
    #derivative wrt cj
    del_Hc_j[0,2] = np.tensordot(del_Rj_inner, SO2_COS_INNER_DERIVATIVE)
    #derivative wrt sj
    del_Hc_j[0,3] = np.tensordot(del_Rj_inner, SO2_SIN_INNER_DERIVATIVE)

    del_Hc_i = np.zeros((1,4))
    #derivative wrt xi, yi
    del_Hc_i[0,:2] = -2*beta*del_pos
    
    #derivative wrt ci
    #first term is from rotation loss, second is from translation
    trans_outer = np.outer(xi[:2]-xj[:2] + RI @ meas[:2,0],meas[:2,0])
    del_Ri_inner = 2*alpha*(RI - RIJ.T @ RJ) + 2*beta*trans_outer
                                  
    del_Hc_i[0,2] = np.tensordot(del_Ri_inner , SO2_COS_INNER_DERIVATIVE)
    #derivative wrt si
    del_Hc_i[0,3] = np.tensordot(del_Ri_inner, SO2_SIN_INNER_DERIVATIVE)
    return loss, del_Hc_i, del_Hc_j

def H_graph_conv(state, meas,alpha=1.0,beta=1.0):
    loss = 0.0
    delta = np.zeros((state.shape))
    for m in meas:
        if m[0] == 'b':
            l, dHi, dHj =  H_between2_conv(state[:,m[1]],state[:,m[2]],meas[m],alpha,beta)
            loss += l
            delta[:,m[1]] += dHi[0,:]
            delta[:,m[2]] += dHj[0,:]
    return loss, delta


def solve_graph_convex(x0, measC, alpha=1.0, beta=1.0, n_steps = 1000, return_states=True):

    ybound = np.inf
    xsh = x0.shape
    lam = 1.0
    losses = []
    lams = []

    alpha=1.0
    beta=1.0

    if return_states:
        states = [x0]
    print(f"\nRelaxed: Optimizing {x0.shape[1]} poses with {len(measC.keys())} constraints\n")
    pbar = tqdm.trange(2000)
    for i in pbar:
        loss, grad = H_graph_conv(x0, measC)

        #don't let constant term move 
        grad[:,0] = 0.0
        losses.append(loss)
        update_state = True
        while(update_state):

            if lam < 1e-20:
                break
            #I am taking the lambda term outside of the pinv as opposed to normal LMQ (so we decrease lambda on a failure) since I find it more natural to formulate the objective function this way.

            upd = x0 - (np.linalg.pinv(grad.reshape((1,-1))) * lam*loss).reshape(xsh)
            #upd = x0 - lam*grad

            ybound = H_graph_conv(upd, measC)[0]

            #print(f"Tried lam {lam}, {ybound} vs {loss}")
            if ybound >= loss:
                lam *= 1e-1
            else:
                update_state = False
                lams.append(lam)
        pbar.set_description(f"{loss:.3f}")
        lam *= 2
        #print("succeeded")

        x0 = upd
        if return_states:
            states.append(np.copy(x0))
            
    print("")

    if return_states:
        return x0, losses, lams, states
    else:
        return x0, losses, lams

def solve_graph(x0,meas,CINV=np.identity(3),num_steps=1000,return_states=True):
    ybound = np.inf
    xsh = x0.shape
    lam = 1.0
    losses = []
    lams = []

    print(f"\nLMQ: Optimizing {x0.shape[1]} poses with {len(measC.keys())} constraints\n")

    
    if return_states:
        states = [x0]
    for i in tqdm.trange(num_steps):
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
        if return_states:
            states.append(np.copy(x0))

    print("")

    if return_states:
        return x0, losses, lams, states
    else:
        return x0, losses, lams
