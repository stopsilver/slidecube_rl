## NNet_SlideCube_statevalue_model_test

import numpy as np
from tensorflow import keras
from slidecube_env import CubeEnv

SN=2
cube_env=CubeEnv(SN)

use_weight=False
if use_weight :
    postfix="weighted"
else :
    postfix="pure"
modelfilename="slcube_stval_"+str(SN)+"x"+str(SN)+"_"+postfix+".h5"

## load NNet
model_val=keras.models.load_model(modelfilename)

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

scramble_depth=20
probs=[1/len(cube_env.init_state_list)]*len(cube_env.init_state_list)
test_num=20
x=np.zeros((0,np.prod(cube_env.encoded_shape)))
for i in range(20) :
    start_state=cube_env.init_state_list[np.random.choice(np.arange(len(cube_env.init_state_list)),p=probs)]
    start_state=np.ravel(start_state)
    data=cube_env.scramble_cube(scramble_depth,
                                    cube_env.ConvertToState(start_state),
                                    include_initial=True)
    for _, s in data:
        enc_s = encode_states(cube_env, [s])
        x=np.concatenate([x,np.reshape(enc_s,[1,-1])],axis=0)

actval,stval=model_val.predict(x)

np.savetxt('cubestval.txt',np.reshape(stval,[-1,scramble_depth+1]))