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

V=np.zeros((numstates))

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

def GetReturnEstimation(idx,deep) :
    # generate ramdom route
    st=states[idx]
    st_list=[]
    for i in range(deep) :
        st_list.append(square_env.transform(st,square_env.sample_action()))
    # return estimation
    G=V[idx_states[st_list[-1]]]
    for pos in reversed(st_list[0:-1]) :
        G=G*gamma+square_env.state_cost(pos)
    return G

gamma=0.9

alpha=0.5
NumRoutes=20
deep=4

for k in range(100) :            # iteration
    Vnew=V.copy()
    for sidx in range(numstates) :         # for all states
        s=0
        for j in range(NumRoutes) :
            G=GetReturnEstimation(sidx,deep)
            s+=G
        s=s/NumRoutes
        Vnew[sidx]=Vnew[sidx]*(1-alpha)+s*alpha
    V=Vnew

    print('k='+str(k+1))
    print(V[0:10])

# save state-value function to 
fid = open("statevalue_"+str(N)+"x"+str(N)+".txt", "w")
np. savetxt(fid,V)
fid.close()