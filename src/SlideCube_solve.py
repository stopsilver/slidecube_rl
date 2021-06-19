## SlideCube_solve.py

import numpy as np
from tensorflow import keras
from slidecube_env import CubeEnv

SN=2
cube_env=CubeEnv(SN)

gamma=1

ep=0.

use_weight=False
if use_weight :
    postfix="weighted"
else :
    postfix="pure"
modelfilename="slcube_stval_"+str(SN)+"x"+str(SN)+"_"+postfix+".h5"

## load NNet
model_val=keras.models.load_model(modelfilename)

## scramble
s=cube_env.scramble_cube(4)
state=s[-1][1]
# print initial state
print(cube_env.render(state))

np.savetxt('startpos.txt',state.sq_pos)

## solve

def encode_states(env, states):
    assert isinstance(env, CubeEnv)
    assert isinstance(states, (list, tuple))

    # states could be list of lists or just list of states
    if isinstance(states[0], list):
        encoded = np.zeros((len(states), len(states[0])) + env.encoded_shape, dtype=np.float32)

        for i, st_list in enumerate(states):
            for j, state in enumerate(st_list):
                encoded[i, j]=env.encode_inplace(state)
    else:
        encoded = np.zeros((len(states), ) + env.encoded_shape, dtype=np.float32)
        for i, state in enumerate(states):
            encoded[i]=env.encode_inplace(state)

    return encoded

def GetBestAction(env,s) :
    states, goals = env.explore_state(s)
    idx=[i for i, x in enumerate(goals) if x]
    if len(idx)>0 :
        return idx[0]

    # Get score from states
    enc_s = encode_states(cube_env, states)
    shape=np.shape(enc_s)
    _,V=model_val.predict(np.reshape(enc_s,[shape[0],-1]))
    # R=-1
    # G=V*gamma+R
    V=np.ravel(V)

    return np.argmax(V)

def GetEpsPolicyAction(env,act_idx) :
    p=np.random.choice([True,False],p=[1-ep,ep])
    if p : return act_idx
    n=len(env.action_enum)
    idx=np.random.choice(np.arange(n,dtype=int),p=np.ones(n)/n)
    return idx

solmoves=[]

cnt=0
while not cube_env.is_goal(state) :
    act_idx=GetBestAction(cube_env,state)
    if ep>0 :   # follow epsilon policy
        act_idx=GetEpsPolicyAction(cube_env,act_idx)
    solmoves.append(act_idx)
    state=cube_env.transform(state,cube_env.action_enum[act_idx])
    print(cube_env.render(state))
    cnt+=1
    if cnt>50 :
        print("Fail!")
        break

np.savetxt('solutionmoves.txt',solmoves)

print("done!")
