# TD0_train.py

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

import slidesquares

slidesquare_type='square4x4'

square_env=slidesquares.get(slidesquare_type)

# model
model = Sequential()
#model.add(Dense(32, activation='relu',input_shape=np.prod(np.asarray(square_env.encoded_shape))))
model.add(Dense(32, activation='relu',input_shape=(np.prod(np.asarray(square_env.encoded_shape)),)))
# model.add(Dense(len(square_env.action_enum), activation='softmax'))
model.add(Dense(1))

model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.Adam())

# training
a=square_env.scramble_square(20)    # initial state
state=a[-1][1]

alpha=0.01
gamma=0.9

while 1 :
    curr_state_enc=np.reshape(
            np.array(np.ravel(square_env.encode_inplace(state))),
        (1,-1))
    # current state-value
    V=model.predict(curr_state_enc)
    # arbitrary move (later must be move defined by policy)
    action=square_env.sample_action()
    new_state=square_env.transform(state,action)
    # compute reward (difference of state costs)
    # new_state_cost=square_env.state_cost(new_state)
    R=square_env.state_cost(state)
    # next state_value
    new_state_enc=np.reshape(
            np.array(np.ravel(square_env.encode_inplace(new_state))),
        (1,-1))
    V1=model.predict(new_state_enc)
    # updated state estimation
    Vt=V+alpha*(gamma*R+V1)
    # train net
    model.fit(curr_state_enc,Vt,epochs=1)
    # update states
    state=new_state
