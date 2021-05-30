## SlSqNet.py

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def SlSqNet(input_shape,
            L1_num,L2_num,A1_num,V1_num,
            Aout_num) :
        
    inputs=Input(shape=input_shape)
    x=Dense(L1_num,activation='relu')(inputs)
    x=Dense(L2_num,activation='relu')(x)
    
    a=Dense(A1_num,activation='relu')(x)
    a=Dense(Aout_num,name='action')(a)

    v=Dense(V1_num,activation='relu')(x)
    v=Dense(1,name='state')(v)

    model=Model(inputs,[a,v],name='SlSqNet')

    return model
