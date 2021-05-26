# SlideSquare_DP_statevalue.py

import numpy as np
from slidesquare_env import SquareEnv

square_env=SquareEnv(3)

## generate list of all states
N=square_env.encoded_shape[0]

from gen_all_pos import gen_all_pos
allpos_list=gen_all_pos(N)
numstates=len(allpos_list)

V=np.random.randn((numstates))
pl=np.zeros((numstates),dtype=np.int)

states={}
idx_states={}
for i in range(numstates) :
    a=allpos_list[i]
    rn=square_env.ConvertToState(a.astype(int))
    states[i]=rn
    idx_states[rn]=i

def GetReward(st0,st1) :
    if square_env.is_goal(st1) : r=0
    else : r=-1
    return r

def GetNewStateAndReward(idx,Move) :
    # move to new state
    st=states[idx]
    if square_env.is_goal(st) : return idx_states[st],0
    st1=square_env.transform(st,square_env.action_enum[Move])
    # reward
    R=GetReward(st,st1)
    return idx_states[st1], R

def GetReturnEstimation(idx,j) :
    sidx1,R=GetNewStateAndReward(idx,j)
    SV1=V[sidx1]
    return (R+gamma*SV1)

gamma=0.9

while True :
    
    ### policy evaluation (value iteration)
    for k in range(10) :            # iteration
        Vnew=V.copy()
        for sidx in range(numstates) :         # for all states
            # s=0
            # for j in range(len(square_env.action_enum)) :
            #     s=s+GetReturnEstimation(sidx)
            # s=s/len(square_env.action_enum)

            s=GetReturnEstimation(sidx,pl[sidx])
            Vnew[sidx]=s # Vnew[sidx]*(1-alpha)+s*alpha
        V=Vnew

        print('k='+str(k+1))
        print(V[0:10])

    ### policy improvement
    pl_old=pl.copy()
    for sidx in range(numstates) :
        st=states[sidx]
        if not square_env.is_goal(st) :
            # next_states=[square_env.transform(st,square_env.action_enum[Move])  for Move in range(len(square_env.action_enum))]
            v=[GetReturnEstimation(sidx,Move) for Move in range(len(square_env.action_enum))]
            pl[sidx]=np.argmax(v)

    if np.array_equal(pl_old,pl) : break

# save state-value function to 
fid = open("statevalue_"+str(N)+"x"+str(N)+".txt", "w")
np.savetxt(fid,V)
fid.close()

fid = open("policy_"+str(N)+"x"+str(N)+".txt", "w")
np.savetxt(fid,pl)
fid.close()