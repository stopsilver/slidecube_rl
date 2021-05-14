"""
Generic square environment
"""
import itertools
import numpy as np
import collections
import copy
import random

State_type = collections.namedtuple("State", field_names=['sq_pos'])
Action_Type = collections.namedtuple("Action", field_names=['actidx','actstr'])

class SquareEnv:
    def __init__(self, N):

        self.N=N
        self.initial_state = State_type(sq_pos=tuple(itertools.chain.from_iterable(itertools.repeat(x, N) for x in range(N))))
        self.goal_cost=self.state_cost(self.initial_state)

        ### Global Init
        j=0
        self.square_move_set=[[None for i in range(N)] for j in range(2*N*(N-1))]
        # rotation in X
        for layer in range(N) :
            for mv in range(N-1) :
                to_idx=layer*N+np.arange(N)
                from_idx=np.roll(to_idx,mv+1)
                for i in range(N) :
                    self.square_move_set[j][i]=(int(from_idx[i]),int(to_idx[i]))
                j=j+1
        # rotation in Y
        for layer in range(N) :
            for mv in range(N-1) :
                to_idx=layer+N*np.arange(N)
                from_idx=np.roll(to_idx,mv+1)
                for i in range(N) :
                    self.square_move_set[j][i]=(int(from_idx[i]),int(to_idx[i]))
                j=j+1

        # Action mnemonics (for convenience)
        Action=[None for i in range((N-1)*N*2)]
        self._inverse_action={}
        k=0
        for i in range(N) :
            for j in range(1,N) :
                s1='X_'+str(i).zfill(2)+'_'+str(j+1).zfill(2)
                s2='X_'+str(i).zfill(2)+'_'+str(N-(j+1)).zfill(2)
                # k=j+(N-1)*i
                Action[k]=Action_Type(actidx=k,actstr=s1)
                k=k+1
                self._inverse_action[s1]=s2
        for i in range(N) :
            for j in range(1,N) :
                s1='Y_'+str(i).zfill(2)+'_'+str(j+1).zfill(2)
                s2='Y_'+str(i).zfill(2)+'_'+str(4*N-(j+1)).zfill(2)
                # k=j+(4*N-1)*i+((4*N-1)*N)
                Action[k]=Action_Type(actidx=k,actstr=s1)
                self._inverse_action[s1]=s2
                k=k+1

        self.action_enum=Action

        self.encoded_shape = (N, N*N)

        # colors (for visualization)
        self.sq_colors = ['W','R','G','B']   # currently upto 4x4
### /Global Init


        # self.name = name
        # self._state_type = state_type
        # self._action_type = action_type
        # self.initial_state = initial_state
        # self._is_goal_pred = is_goal_pred
        # self.action_enum = action_enum
        # self._transform_func = transform_func
        # self._inverse_action_func = inverse_action_func
        # self._same_layer_action = same_layer_action
        # self._render_func = render_func
        # self.encoded_shape = encoded_shape
        # self._encode_func = encode_func
        # self._state_cost_func=state_cost_func
        # self._convert_to_state_func=convert_to_state_func

    # def __repr__(self):
    #     return "SquareEnv(%r)" % self.name

    # wrapper functions
    def is_goal(self, state):
        assert isinstance(state, State_type)
#        return state.sq_pos == initial_state.sq_pos
        return self.state_cost(state)==self.goal_cost

    def transform(self, state, action):
        assert isinstance(state, State_type)
        assert isinstance(action, Action_Type)

        def _permute(t, m):
            """
            Perform permutation of tuple according to mapping m
            """
            r = list(t)
            for from_idx, to_idx in m:
                r[to_idx] = t[from_idx]
            return r

        act_num=action[0]
        a=_permute(state.sq_pos,self.square_move_set[act_num])

        return State_type(sq_pos=tuple(a))

    def inverse_action(self, action):
        assert isinstance(action, Action_Type)
        return self._inverse_action[action]

    def render(self, state):
        assert isinstance(state, State_type)
        a=state.sq_pos
        r=''
        k=0
        for i in range(self.N) :
            for j in range(self.N) :
                r=r+self.sq_colors[a[k]]
                k=k+1
            r=r+'\n'
        return r

    def encode_inplace(self, state):
        """
        Encode square into existig zeroed numpy array
        Follows encoding described in paper https://arxiv.org/abs/1805.07470
        :param target: numpy array
        :param state: state to be encoded
        """
        assert isinstance(state, State_type)
        target=np.zeros(self.encoded_shape)
        a=state.sq_pos
        for pos in range(len(a)):
            target[a[pos], pos] = 1
        return target

    # Utility functions
    def _same_layer_action(self,mov1,mov2):
        M=self.N-1
        if int(mov1[0]/M)==int(mov2[0]/M) :
            return 1
        return 0

    def sample_action(self, prev_action=None):
        while True:
            res = self.action_enum[random.randrange(len(self.action_enum))]
            # if prev_action is None or self.inverse_action(res) != prev_action:
            #     return res
            if prev_action is None :
                return res  
            if self._same_layer_action(res,prev_action) == False :
                return res            

    def scramble(self, actions):
        s = self.initial_state
        for action in actions:
            s = self.transform(s, action)
        return s

    def is_state(self, state):
        return isinstance(state, self._state_type)

    def scramble_square(self, scrambles_count, return_inverse=False, include_initial=False):
        """
        Generate sequence of random square scrambles
        :param scrambles_count: count of scrambles to perform
        :param return_inverse: if True, inverse action is returned
        :return: list of tuples (depth, state[, inverse_action])
        """
        assert isinstance(scrambles_count, int)
        assert scrambles_count > 0

        state = self.initial_state
        result = []
        if include_initial:
            assert not return_inverse
            result.append((1, state))
        prev_action = None
        for depth in range(scrambles_count):
            action = self.sample_action(prev_action=prev_action)
            state = self.transform(state, action)
            prev_action = action
            if return_inverse:
                inv_action = self.inverse_action(action)
                res = (depth+1, state, inv_action)
            else:
                res = (depth+1, state)
            result.append(res)
        return result

    def explore_state(self, state):
        """
        Expand square state by applying every action to it
        :param state: state to explore
        :return: tuple of two lists: [states reachable], [flag that state is initial]
        """
        res_states, res_flags = [], []
        for action in self.action_enum:
            new_state = self.transform(state, action)
            is_init = self.is_goal(new_state)
            res_states.append(new_state)
            res_flags.append(is_init)
        return res_states, res_flags

    def state_cost(self, state):
        """
        Estimates cost of state from viewpoint of squares mutual arrangement
        """
        N=self.N

        def GetHorizontalAlingment(a) :
            r=[i for i, j in zip(a[0:-1], a[1:]) if i == j]
            return len(r)

        assert isinstance(state, State_type)
        a=state.sq_pos
        s=0
        for i in range(N) :
            s=s+GetHorizontalAlingment(a[i*N:(i+1)*N])
        return s

    def ConvertToState(self,a):
        return State_type(sq_pos=tuple(a))


# def register(square_env):
#     assert isinstance(square_env, SquareEnv)
#     global _registry

#     if square_env.name in _registry:
#         log.warning("Square environment %s is already registered, ignored", square_env)
#     else:
#         _registry[square_env.name] = square_env


# def get(name):
#     assert isinstance(name, str)
#     return _registry.get(name)


# def names():
#     return list(sorted(_registry.keys()))
