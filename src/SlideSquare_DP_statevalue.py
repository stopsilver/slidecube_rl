# SlideSquare_DP_statevalue.py

import numpy as np
from slidesquare_env import SquareEnv

# slidesquare_type='square4x4'   # in reality "2x2"

# square_env=slidesquares.get(slidesquare_type)

square_env=SquareEnv(3)

## generate list of all states
N=square_env.encoded_shape[0]

from gen_all_pos import gen_all_pos
allpos_list=gen_all_pos(N)
numstates=len(allpos_list)

states={}
idx_states={}
for i in range(numstates) :
    a=allpos_list[i]
    rn=square_env.ConvertToState(a.astype(int))
    states[i]=rn
    idx_states[rn]=i

## render all states
# for key in states.keys() :
#     print(square_env.render(states[key]))

## interesting idea from (https://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary)
# mydict = {'george': 16, 'amber': 19}
# print(list(mydict.keys())[list(mydict.values()).index(16)])  # Prints george

def GetNewStateAndReward(idx,Move) :
    # move to new state
    st=states[idx]
    st1=square_env.transform(st,square_env.action_enum[Move])
    # reward
    cost=square_env.state_cost(st)
    # cost1=square_env.state_cost(st1)
    # R=cost1-cost
    R=cost
    R=max(R,0)
    return idx_states[st1], R

V=np.zeros((numstates))

gamma=0.9

for k in range(100) :            # iteration
    Vnew=np.zeros((numstates))
    for sidx in range(numstates) :         # for all states
        s=0
        for j in range(len(square_env.action_enum)) :
            sidx1,R=GetNewStateAndReward(sidx,j)
            SV1=V[sidx1]
            s=s+(R+gamma*SV1)

        s=s/len(square_env.action_enum)
        Vnew[sidx]=s

    V=Vnew

    print('k='+str(k+1))
    print(V[0:10])

# save state-value function to 
fid = open("statevalue_"+str(N)+"x"+str(N)+".txt", "w")
np. savetxt(fid,V)
fid.close()