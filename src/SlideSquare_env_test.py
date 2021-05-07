# SlideSquare_env_test.py

import numpy as np

from slidesquare_env import SquareEnv

# slidesquare_type='square4x4'

square_env=SquareEnv(4)

rn=square_env.render(square_env.initial_state)
print(rn)

# a=square_env.transform(square_env.initial_state,square_env.action_enum[23])
# rn=square_env.render(a)
# print(rn)

# b=square_env.encode_inplace(a)
# print(b)
# print(' ')

# a=square_env.scramble((square_env.action_enum[23],square_env.action_enum[0]))
# rn=square_env.render(a)
# print(rn)

# a=square_env.scramble_square(20)
# rn=square_env.render(a[-1][1])
# print(rn)
# print(square_env.state_cost(a[-1][1]))

# rn=square_env.render(square_env.initial_state)
# print(rn)
# print(square_env.state_cost(square_env.initial_state))

# a=square_env.transform(square_env.initial_state,square_env.action_enum[23])
# rn=square_env.render(a)
# print(rn)
# print(square_env.state_cost(a))