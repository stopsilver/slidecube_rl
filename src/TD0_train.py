# TD0_train.py

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from slidesquare_env import SquareEnv

square_env=SquareEnv(2)

# model
model = Sequential()
#model.add(Dense(32, activation='relu',input_shape=np.prod(np.asarray(square_env.encoded_shape))))
model.add(Dense(8, activation='relu',input_shape=(np.prod(square_env.encoded_shape),)))
# model.add(Dense(len(square_env.action_enum), activation='softmax'))
model.add(Dense(1))

model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.RMSprop(lr=3e-3))

# initial scramble
a=square_env.scramble_square(20)    # initial state
state=a[-1][1]

alpha=0.05
gamma=0.9

## init buffer
x_train_buf=np.zeros((0,np.prod(square_env.encoded_shape)))
y_train_buf=np.zeros((0,))
Buf_Capacity=40              # buf max len

def prepare_training_data(st_list) :
    state_list=[s[1] for s in st_list]
    rewards=[square_env.state_cost(s) for s in state_list[1:]]
    # current state-values
    encoded_states=np.zeros((len(state_list),np.prod(square_env.encoded_shape)))
    for i in range(len(state_list)) : encoded_states[i,:]=np.ravel(square_env.encode_inplace(state_list[i]))
    v_list=model.predict(encoded_states)
    v_list=np.squeeze(v_list)
    # prepare training data
    x_train=encoded_states[0:-1]
    y_train=(1-alpha)*v_list[0:-1]+alpha*(rewards+gamma*v_list[1:])
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

## training
N=10    # new moves
cnt=0
while 1 :
    st_list = square_env.scramble_square(N, include_initial=state)
    x_train, y_train = prepare_training_data(st_list)
    state=st_list[-1]
    Fill_Buf(x_train, y_train)
    model.fit(x_train_buf,y_train_buf,epochs=10,verbose=3)
    cnt+=1
    if cnt>100 : break

print("done!")

# display
a=[ [0,0,1,1],
    [0,1,0,1],
    [0,1,1,0],
    [1,0,0,1],
    [1,0,1,0],
    [1,1,0,0]
    ]
encs=np.zeros((6,8))
for i in range(len(a)) :
    encs[i,:]=np.ravel(square_env.encode_inplace(square_env.ConvertToState(a[i])))

v_list=model.predict(encs)

print(v_list)
