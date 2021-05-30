# TD0_train.py

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from SlSqNet import SlSq_val, SlSq_act
import random

from slidesquare_env import SquareEnv

SN=3
HLN=32
L1_num=128
L2_num=64
A1_num=32
V1_num=32

gamma=0.9

square_env=SquareEnv(SN)

# model for state value
model_val=SlSq_val(int(np.prod(square_env.encoded_shape)),
                L1_num,L2_num,V1_num)
loss_val = "mean_squared_error"
model_val.compile(loss=loss_val, optimizer=keras.optimizers.RMSprop(lr=1e-3))

# model for action value
model_act=SlSq_act(int(np.prod(square_env.encoded_shape)),
                L1_num,L2_num,A1_num,
                len(square_env.action_enum))
loss_act = "categorical_crossentropy"
model_act.compile(loss=loss_act, optimizer=keras.optimizers.RMSprop(lr=1e-3))


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

## initial states
from itertools import permutations
init_state_list=[]
perms=permutations([i for i in range(SN)])
for a in perms:
    init_state_list.append(square_env.ConvertToState([[i]*SN for i in a]))
probs=[1/len(init_state_list)]*len(init_state_list)

def make_pilot_data_buffer(env, buf_size, scramble_depth):
    """
    Create data buffer with scramble states and explored substates
    :param env: env to use
    :param buf_size: how many states to generate
    :param scramble_depth: how deep to scramble
    :return: list of tuples
    """
    result = []
    data = []
    rounds = buf_size // scramble_depth
    for _ in range(rounds):
        start_state=init_state_list[np.random.choice(np.arange(len(init_state_list)),p=probs)]
        start_state=np.ravel(start_state)
        data.extend(env.scramble_square(scramble_depth,
                                        env.ConvertToState(start_state),
                                        include_initial=True))

    # explore each state
    for depth, s in data:
        states, goals = env.explore_state(s)
        enc_s = encode_states(env, [s])
        enc_states = encode_states(env, states)
        result.append((enc_s, depth, env.is_goal(s), enc_states, goals))
    return result

def prepare_state_batch(env, model_val, model_act, buf_size, scramble_depth):
    """
    Sample batch of given size from scramble buffer produced by make_scramble_buffer
    :param scramble_buffer: scramble buffer
    :param model: network to use to calculate targets
    :param batch_size: size of batch to generate
    :param value_targets: targets
    :return: tensors
    """
    data = make_pilot_data_buffer(env, buf_size, scramble_depth)
    states, depths, is_goals, explored_states, explored_goals = zip(*data)

    # handle visited states
    states = np.stack(states)
    shape = states.shape
    act_t = model_act.predict(np.reshape(states,(shape[0], shape[2]*shape[3])))
    act_t=np.argmax(act_t,axis=-1)

    # handle explored states
    explored_states = np.stack(explored_states)
    shape = explored_states.shape
    
    value_t = model_val.predict(np.reshape(explored_states,(shape[0]*shape[1], shape[2]*shape[3])))
    value_t = value_t.reshape(shape[0], shape[1])
    value_train_t=value_t[range(shape[0]),act_t]
    R=-1
    value_train_t = value_train_t * gamma + R
    value_train_t=np.reshape(value_train_t,(shape[0],1))

    # force goal states to get 0
    explored_goals=np.stack(explored_goals)
    explored_goals=explored_goals[range(shape[0]),act_t]
    for i in range(len(explored_goals)) :
        if explored_goals[i]==True :
           value_train_t[i]=R
        if is_goals[i]==True :
           value_train_t[i]=0

    enc_input = np.reshape(np.stack(states),(shape[0], shape[2]*shape[3]))
    
    return enc_input, value_train_t

def prepare_action_batch(env, model_val, model_act, buf_size, scramble_depth):
    """
    Sample batch of given size from scramble buffer produced by make_scramble_buffer
    :param scramble_buffer: scramble buffer
    :param model: network to use to calculate targets
    :param batch_size: size of batch to generate
    :param value_targets: targets
    :return: tensors
    """
    data = make_pilot_data_buffer(env, buf_size, scramble_depth)
    states, depths, is_goals, explored_states, explored_goals = zip(*data)

    # handle explored states
    explored_states = np.stack(explored_states)
    shape = explored_states.shape
    
    value_t = model_val.predict(np.reshape(explored_states,(shape[0]*shape[1], shape[2]*shape[3])))
    value_t = value_t.reshape(shape[0], shape[1])
    R=-1
    value_t = value_t * gamma + R
    value_t=np.reshape(value_t,(shape[0],shape[1]))

    max_act_t = keras.utils.to_categorical(np.argmax(value_t, axis=-1))

    enc_input = np.reshape(np.stack(states),(shape[0], shape[2]*shape[3]))
    # max_act_t=np.eye(shape[1])[max_act_t]
    
    return enc_input, max_act_t

def store_data(env,model_val,model_act) :
    from gen_all_pos import gen_all_pos

    # store data
    all_pos_list=gen_all_pos(SN)

    encs=np.zeros((len(all_pos_list),SN*SN*SN))
    for i in range(len(all_pos_list)) :
        encs[i,:]=np.ravel(env.encode_inplace(env.ConvertToState(all_pos_list[i].astype(int))))

    v_list=model_val.predict(encs)

    fid=open("sl_"+str(SN)+"x"+str(SN)+".txt","w")
    np.savetxt(fid,v_list)
    fid.close()

    model_val.save("sl_stval_"+str(SN)+"x"+str(SN)+".h5")

    model_act.save("sl_act_"+str(SN)+"x"+str(SN)+".h5")

## init buffer
train_batch_size=250
train_buf_size=1000
train_scramble_depth=10

cnt=0
while 1 :
    for i in range(10) :
        x_t, y_value_t = prepare_state_batch(square_env, model_val,model_act,train_buf_size,train_scramble_depth)
        print("i = "+str(i)+". Pushed new data in train buffer")
        model_val.fit(x_t,y_value_t,batch_size=train_batch_size,epochs=10,verbose=1)

    for k in range(10) :
        x_t, y_policy_t = prepare_action_batch(square_env, model_val,model_act,train_buf_size,train_scramble_depth)
        print("k = "+str(k)+". Pushed new data in train buffer")
        model_act.fit(x_t,y_policy_t,batch_size=train_batch_size,epochs=10,verbose=1)

    cnt+=1
    print("==> cnt = "+str(cnt))

    if cnt % 10 ==0 :
        store_data(square_env,model_val,model_act)
        print("model stored")

    if cnt>100 : break

print("done!")

store_data(square_env,model_val,model_act)
print("model stored")