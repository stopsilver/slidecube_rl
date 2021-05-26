## SlideSquare_solve.py

import numpy as np
from slidesquare_env import SquareEnv
from gen_all_pos import gen_all_pos

## read state-value
V=np.loadtxt('statevalue_3x3.txt')
pl=np.loadtxt('policy_3x3.txt')
pl=pl.astype(int)

## read all positions enumerated
all_pos_list=gen_all_pos(3)

## init environment
square_env=SquareEnv(3)

## scramble
s=square_env.scramble_square(20)
state=s[-1][1]
# print initial state
print(square_env.render(state))

## solve
# def GetScore(s) :
#     global V
#     global all_pos_list
#     a=[(x==s.sq_pos).all() for x in all_pos_list]
#     idx=a.index(True)
#     return V[idx]

# def GetReward(st0,st1) :
#     if square_env.is_goal(st1) : r=0
#     else : r=-1
#     return r

# def GetBestAction(env,s) :
#     # global V
#     G=[]
#     gamma=0.9
#     for i in range(len(env.action_enum)) :
#         s1=env.transform(s,env.action_enum[i])
#         # r=env.state_cost(s1)
#         r=GetReward(s,s1)
#         V1=GetScore(s1)
#         G.append(r+gamma*V1)
#     return G.index(max(G))

while not square_env.is_goal(state) :
    # act_idx=GetBestAction(square_env,state)
    idx=[i for i, x in enumerate(all_pos_list) if (x == state).all()]
    act_idx=pl[idx[0]]
    state=square_env.transform(state,square_env.action_enum[act_idx])
    print(square_env.render(state))

print("done!")
