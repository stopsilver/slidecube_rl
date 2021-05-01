import itertools
import numpy as np
import collections
import copy

from . import _env
from . import _common

State = collections.namedtuple("State", field_names=['sq_pos'])
Action_Type = collections.namedtuple("Action", field_names=['actidx','actstr'])

N=4
initial_state = State(sq_pos=tuple(itertools.chain.from_iterable(itertools.repeat(x, N) for x in range(N))))

### Global Init
j=0
square_move_set=[[None for i in range(N)] for j in range(2*N*(N-1))]
# rotation in X
for layer in range(N) :
    for mv in range(N-1) :
        to_idx=layer*N+np.arange(N)
        from_idx=np.roll(to_idx,mv+1)
        for i in range(N) :
            square_move_set[j][i]=(int(from_idx[i]),int(to_idx[i]))
        j=j+1
# rotation in Y
for layer in range(N) :
    for mv in range(N-1) :
        to_idx=layer+N*np.arange(N)
        from_idx=np.roll(to_idx,mv+1)
        for i in range(N) :
            square_move_set[j][i]=(int(from_idx[i]),int(to_idx[i]))
        j=j+1

# Action mnemonics (for convenience)
Action=[None for i in range((N-1)*N*2)]
_inverse_action={}
k=0
for i in range(N) :
    for j in range(1,N) :
        s1='X_'+str(i).zfill(2)+'_'+str(j+1).zfill(2)
        s2='X_'+str(i).zfill(2)+'_'+str(N-(j+1)).zfill(2)
        # k=j+(N-1)*i
        Action[k]=Action_Type(actidx=k,actstr=s1)
        k=k+1
        _inverse_action[s1]=s2
for i in range(N) :
    for j in range(1,N) :
        s1='Y_'+str(i).zfill(2)+'_'+str(j+1).zfill(2)
        s2='Y_'+str(i).zfill(2)+'_'+str(4*N-(j+1)).zfill(2)
        # k=j+(4*N-1)*i+((4*N-1)*N)
        Action[k]=Action_Type(actidx=k,actstr=s1)
        _inverse_action[s1]=s2
        k=k+1

### /Global Init

def SameLayerAction(mov1,mov2) :
    M=N-1
    if int(mov1[0]/M)==int(mov2[0]/M) :
        return 1
    return 0

def is_initial(state):
    assert isinstance(state, State)
    return state.sq_pos == initial_state.sq_pos

def inverse_action(action):
    assert isinstance(action, Action)
    return _inverse_action[action]

def transform(state, action):
    assert isinstance(state, State)
    assert isinstance(action, Action_Type)

    act_num=action[0]
    a=_common._permute(state.sq_pos,square_move_set[act_num])

    return State(sq_pos=tuple(a))


# colors (for visualization)
sq_colors = ['W','R','G','B']


def render_razv(state):
    assert isinstance(state, State)
    a=state.sq_pos
    r=''
    k=0
    for i in range(N) :
        for j in range(N) :
            r=r+sq_colors[a[k]]
            k=k+1
        r=r+'\n'
    return r

encoded_shape = (4, 16)

def encode_inplace(state):
    """
    Encode square into existig zeroed numpy array
    Follows encoding described in paper https://arxiv.org/abs/1805.07470
    :param target: numpy array
    :param state: state to be encoded
    """
    assert isinstance(state, State)
    target=np.zeros(encoded_shape)
    a=state.sq_pos
    for pos in range(len(a)):
        target[a[pos], pos] = 1
    return target

def state_cost_estimation(state) :
    """
    Estimates cost of state from viewpoint of squares mutual arrangement
    """
    def GetHorizontalAlingment(a) :
        r=[i for i, j in zip(a[0:-1], a[1:]) if i == j]
        return len(r)

    assert isinstance(state, State)
    a=state.sq_pos
    s=0
    for i in range(N) :
        s=s+GetHorizontalAlingment(a[i*N:(i+1)*N])
    return s

# register env
_env.register(_env.SquareEnv(name="square4x4", state_type=State, action_type=Action_Type, initial_state=initial_state,
                           is_goal_pred=is_initial, action_enum=Action,
                           transform_func=transform, inverse_action_func=inverse_action,
                           same_layer_action=SameLayerAction,
                           render_func=render_razv, encoded_shape=encoded_shape, encode_func=encode_inplace,
                           state_cost_func=state_cost_estimation))
