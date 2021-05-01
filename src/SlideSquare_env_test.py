# SlideSquare_env_test.py

import numpy as np

import slidesquares

slidesquare_type='square4x4'

square_env=slidesquares.get(slidesquare_type)

# rn=square_env._render_func(square_env.initial_state)
# print(rn)

# a=square_env._transform_func(square_env.initial_state,square_env.action_enum[23])
# rn=square_env._render_func(a)
# print(rn)

# b=np.zeros(square_env.encoded_shape)
# square_env.encode_inplace(b,square_env.initial_state)
# print(b)
# print(' ')

# b=np.zeros(square_env.encoded_shape)
# square_env.encode_inplace(b,a)
# print(b)
# print(' ')

# a=square_env.scramble((square_env.action_enum[23],square_env.action_enum[0]))
# rn=square_env._render_func(a)
# print(rn)

# a=square_env.scramble_square(20)
# rn=square_env._render_func(a[-1][1])
# print(rn)

# rn=square_env._render_func(square_env.initial_state)
# print(rn)
# print(square_env.state_cost(square_env.initial_state))

a=square_env._transform_func(square_env.initial_state,square_env.action_enum[23])
rn=square_env._render_func(a)
print(rn)
print(square_env.state_cost(a))