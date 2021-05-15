# SlideSquare_DP_statevalue.py

import numpy as np
from slidesquare_env import SquareEnv

square_env=SquareEnv(3)

## generate list of all states
N=square_env.encoded_shape[0]

from gen_all_pos import gen_all_pos
allpos_list=gen_all_pos(N)
numstates=len(allpos_list)

V=np.zeros((numstates))

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
    if square_env.is_goal(st) : return 0,0
    st1=square_env.transform(st,square_env.action_enum[Move])
    # reward
    R=GetReward(st,st1)
    return idx_states[st1], R

def GetReturnEstimation(idx) :
    sidx1,R=GetNewStateAndReward(idx,j)
    SV1=V[sidx1]
    return (R+gamma*SV1)

gamma=0.9

for k in range(100) :            # iteration
    Vnew=V.copy()
    for sidx in range(numstates) :         # for all states
        s=0
        for j in range(len(square_env.action_enum)) :
            s=s+GetReturnEstimation(sidx)

        s=s/len(square_env.action_enum)
        Vnew[sidx]=s # Vnew[sidx]*(1-alpha)+s*alpha
    V=Vnew

    print('k='+str(k+1))
    print(V[0:10])

# save state-value function to 
fid = open("statevalue_"+str(N)+"x"+str(N)+".txt", "w")
np. savetxt(fid,V)
fid.close()