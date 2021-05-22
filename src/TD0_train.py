# TD0_train.py

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from slidesquare_env import SquareEnv

# SN=2
# HLN=8
SN=3
HLN=32

square_env=SquareEnv(SN)

# model
model = Sequential()
model.add(Dense(HLN, activation='relu',input_shape=(np.prod(square_env.encoded_shape),)))
# model.add(Dense(len(square_env.action_enum), activation='softmax'))
model.add(Dense(1))

model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.RMSprop(lr=1e-4))

# initial scramble
# a=square_env.scramble_square(20)    # initial state
# state=a[-1][1]

alpha=0.05
gamma=0.9

## init buffer
x_train_buf=np.zeros((0,np.prod(square_env.encoded_shape)))
y_train_buf=np.zeros((0,))
Buf_Capacity=3000              # buf max len

# # # forward
# def prepare_training_data(st_list) :
#     state_list=[s[1] for s in st_list]
#     encoded_states=np.zeros((len(state_list),np.prod(square_env.encoded_shape)))
#     for i in range(len(state_list)) : encoded_states[i,:]=np.ravel(square_env.encode_inplace(state_list[i]))
#     # compute state_values
#     v_list=model.predict(encoded_states)
#     v_list=np.squeeze(v_list)
#     # prepare training data
#     x_train=encoded_states[0:-1]
#     rewards=[0 if square_env.is_goal(s) else -1  for s in state_list[1:]]
#     y_train=(1-alpha)*v_list[0:-1]+alpha*(rewards+gamma*v_list[1:])
#     # zero value for terminal state
#     for i in range(len(x_train)) :
#         if square_env.is_goal(state_list[i]) :
#             y_train[i]=0
#     return x_train, y_train

# # # backward
def prepare_training_data(st_list) :
    state_list=[s[1] for s in st_list]
    encoded_states=np.zeros((len(state_list),np.prod(square_env.encoded_shape)))
    for i in range(len(state_list)) : encoded_states[i,:]=np.ravel(square_env.encode_inplace(state_list[i]))
    # compute state_values
    v_list=model.predict(encoded_states)
    v_list=np.squeeze(v_list)
    # prepare training data
    x_train=encoded_states[1:]
    rewards=[0 if square_env.is_goal(s) else -1  for s in state_list[0:-1]]
    y_train=(1-alpha)*v_list[1:]+alpha*(rewards+gamma*v_list[0:-1])
    # zero value for terminal state
    for i in range(1,len(state_list)) :
        if square_env.is_goal(state_list[i]) :
            y_train[i-1]=0
    return x_train, y_train

def Fill_Buf(x_train, y_train) :
    global x_train_buf
    global y_train_buf
    L=len(y_train)
    L_buf=len(y_train_buf)
    Rem=L_buf+L-Buf_Capacity
    Rem=max([Rem,0])        ## all negative change to 0
    x_train_buf=np.concatenate([x_train_buf[Rem:],x_train])
    y_train_buf=np.concatenate([y_train_buf[Rem:],y_train])

## terminal states
from itertools import permutations
init_state_list=[]
perms=permutations([i for i in range(SN)])
for a in perms:
    init_state_list.append(square_env.ConvertToState([[i]*SN for i in a]))
probs=[1/len(init_state_list)]*len(init_state_list)

## training
N=20    # new moves
NumTracks=20
cnt=0
while 1 :
    x_train=[]
    y_train=[]
    for i in range(NumTracks) :
        init_state=init_state_list[np.random.choice(np.arange(6),p=probs)]
        st_list = square_env.scramble_square(N, include_initial=init_state)
        x_t, y_t = prepare_training_data(st_list)
        if len(x_train) == 0 :
            x_train=x_t
            y_train=y_t
        else:
            x_train=np.concatenate([x_train,x_t])
            y_train=np.concatenate([y_train,y_t])
    # state=st_list[-1]
    Fill_Buf(x_train, y_train)
    model.fit(x_train_buf,y_train_buf,epochs=10,verbose=1)
    cnt+=1
    print("==> cnt = "+str(cnt))

    if cnt % 100 ==0 :
        # store data
        from gen_all_pos import gen_all_pos
        all_pos_list=gen_all_pos(SN)

        encs=np.zeros((len(all_pos_list),SN*SN*SN))
        for i in range(len(all_pos_list)) :
            encs[i,:]=np.ravel(square_env.encode_inplace(square_env.ConvertToState(all_pos_list[i].astype(int))))

        v_list=model.predict(encs)

        fid=open("sl_"+str(SN)+".txt","w")
        np.savetxt(fid,v_list)
        fid.close()

        model.save("sl_"+str(SN)+".h5")

    if cnt>1000 : break

print("done!")

# store data
from gen_all_pos import gen_all_pos
all_pos_list=gen_all_pos(SN)

encs=np.zeros((len(all_pos_list),SN*SN*SN))
for i in range(len(all_pos_list)) :
    encs[i,:]=np.ravel(square_env.encode_inplace(square_env.ConvertToState(all_pos_list[i].astype(int))))

v_list=model.predict(encs)

fid=open("sl_"+str(SN)+".txt","w")
np.savetxt(fid,v_list)
fid.close()

model.save("sl_"+str(SN)+".h5")