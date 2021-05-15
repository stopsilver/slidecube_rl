## SlideSquare_solve.py

import numpy as np
from slidesquare_env import SquareEnv
from gen_all_pos import gen_all_pos

## read state-value
V=np.loadtxt('statevalue_3x3.txt')

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
def GetScore(s) :
    global V
    global all_pos_list
    # idx=all_pos_list.all(np.array(s.sq_pos))
    a=[(x==s.sq_pos).all() for x in all_pos_list]
    idx=a.index(True)
    return V[idx]

def GetReward(st0,st1) :
    return min(square_env.state_cost(st1)-square_env.state_cost(st0),0)

def GetBestAction(env,s) :
    # global V
    G=[]
    gamma=0.9
    for i in range(len(env.action_enum)) :
        s1=env.transform(s,env.action_enum[i])
        # r=env.state_cost(s1)
        r=GetReward(s,s1)
        V1=GetScore(s1)
        G.append(r+gamma*V1)
    return G.index(max(G))

while not square_env.is_goal(state) :
    act_idx=GetBestAction(square_env,state)
    state=square_env.transform(state,square_env.action_enum[act_idx])
    print(square_env.render(state))

print("done!")
