# SlideSquare_DP_statevalue.py

import numpy as np
import slidesquares

slidesquare_type='square4x4'   # in reality "2x2"

square_env=slidesquares.get(slidesquare_type)

## generate list of all states
# 0
states={}
idx_states={}
rn=square_env.initial_state
states[0]=rn
idx_states[rn]=0
# 1
rn1=square_env._transform_func(rn,square_env.action_enum[2])
states[1]=rn1
idx_states[rn1]=1
# 2
rn2=square_env._transform_func(rn1,square_env.action_enum[3])
states[2]=rn2
idx_states[rn2]=2
# 3
rn3=square_env._transform_func(rn1,square_env.action_enum[0])
states[3]=rn3
idx_states[rn3]=3
# 4
rn4=square_env._transform_func(rn1,square_env.action_enum[1])
states[4]=rn4
idx_states[rn4]=4
# 5
rn5=square_env._transform_func(rn,square_env.action_enum[3])
states[5]=rn5
idx_states[rn5]=5

## render all states
for key in states.keys() :
    print(square_env.render(states[key]))

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

V=np.zeros((6))

gamma=0.9

for k in range(100) :            # iteration
    Vnew=np.zeros((6))
    for sidx in range(6) :         # for all states
        s=0
        for j in range(len(square_env.action_enum)) :
            sidx1,R=GetNewStateAndReward(sidx,j)
            SV1=V[sidx1]
            s=s+(R+gamma*SV1)

        s=s/len(square_env.action_enum)
        Vnew[sidx]=s

    V=Vnew

    print('k='+str(k+1))
    print(V)
