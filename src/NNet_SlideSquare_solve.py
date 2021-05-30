## SlideSquare_solve.py

import numpy as np
from tensorflow import keras
from slidesquare_env import SquareEnv

SN=3

## load policy NNet
model_act=keras.models.load_model("sl_act_"+str(SN)+"x"+str(SN)+".h5")

## init environment
square_env=SquareEnv(3)

## scramble
s=square_env.scramble_square(20)
state=s[-1][1]
# print initial state
print(square_env.render(state))

def encode_states(env, states):
    assert isinstance(env, SquareEnv)
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

while not square_env.is_goal(state) :
    # get move
    # states, _ = square_env.explore_state(state)
    # enc_states = encode_states(square_env, states)
    # shape=enc_states.shape
    # act_t = model_act.predict(np.reshape(enc_states,(shape[0], shape[1]*shape[2])))
    enc_state = encode_states(square_env, [state])
    shape=enc_state.shape
    act_t = model_act.predict(np.reshape(enc_state,(shape[0], shape[1]*shape[2])))
    act_idx=np.argmax(act_t,axis=-1)

    # execute move
    state=square_env.transform(state,square_env.action_enum[np.int(act_idx)])
    print(square_env.render(state))

print("done!")
