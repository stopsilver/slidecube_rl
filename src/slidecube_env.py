"""
Generic Cube environment
"""
import itertools
import numpy as np
import collections
import copy
import random

State_type = collections.namedtuple("State", field_names=['sq_pos'])
Action_Type = collections.namedtuple("Action", field_names=['actidx','actstr'])

class CubeEnv:
    def __init__(self, N):

        self.N=N
        self.initial_state = State_type(sq_pos=tuple(itertools.chain.from_iterable(itertools.repeat(x, N*N) for x in range(6))))
        # self.goal_cost=self.state_cost(self.initial_state)

        ## initial states list
        from itertools import permutations
        self.init_state_list=[]
        perms=permutations([i for i in range(6)])
        for a in perms:
            self.init_state_list.append(self.ConvertToState([[i]*N*N for i in a]))

        xgrid=np.arange(-1,1,2/N)+1/N
        # ygrid=xgrid

        coordref=np.zeros((N,N,3))
        coordref[:,:,1]=[xgrid for i in range(N)]
        coordref[:,:,0]=coordref[:,:,1].transpose()
        coordref[:,:,2]=np.ones((N,N))

        # print(coordref)

        coord=np.zeros((N,N,3,6))
        coord[:,:,:,0]=coordref                                                # up
        coord[:,:,:,1]=coordref[:,:,(0,2,1)]; coord[:,:,1,1]=-coord[:,:,1,1]   # left
        coord[:,:,:,2]=coordref[:,:,(2,0,1)];                                  # front
        coord[:,:,:,3]=coordref[:,:,(0,2,1)];                                  # right
        coord[:,:,:,4]=coordref[:,:,(2,0,1)]; coord[:,:,0,4]=-coord[:,:,0,4]   # back
        coord[:,:,:,5]=coordref;              coord[:,:,2,5]=-coord[:,:,2,5]   # down

        xcoord=np.squeeze(coord[:,:,0,:]).reshape((-1),order='F')
        ycoord=np.squeeze(coord[:,:,1,:]).reshape((-1),order='F')
        zcoord=np.squeeze(coord[:,:,2,:]).reshape((-1),order='F')

        rotplane=np.zeros((N*3,N*4))
        # x rotation plane
        for j in range(N) :                 # all rotation planes in one axis
            idx=np.squeeze(np.argwhere(xcoord==xgrid[j]))
            a=ycoord[idx]; b=zcoord[idx]
            a=np.angle(a+b*1j)
            idx1=np.argsort(a)
            rotplane[j,:]=idx[idx1]

        # y rotation plane
        for j in range(N) :                 # all rotation planes in one axis
            idx=np.squeeze(np.argwhere(ycoord==xgrid[j]))
            a=zcoord[idx]; b=xcoord[idx]
            a=np.angle(a+b*1j)
            idx1=np.argsort(a)
            rotplane[j+N,:]=idx[idx1]

        # z rotation plane
        for j in range(N) :                 # all rotation planes in one axis
            idx=np.squeeze(np.argwhere(zcoord==xgrid[j]))
            a=xcoord[idx]; b=ycoord[idx]
            a=np.angle(a+b*1j)
            idx1=np.argsort(a)
            rotplane[j+N*2,:]=idx[idx1]

        j=0
        self.cube_move_set=[[None for i in range(4*N)] for j in range(3*N*(4*N-1))]
        for layer in range(N*3) :
            for mv in range(4*N-1) :
                to_idx=rotplane[layer,:]
                from_idx=np.roll(rotplane[layer,:],mv+1)
                for i in range(4*N) :
                    self.cube_move_set[j][i]=(int(from_idx[i]),int(to_idx[i]))
                j=j+1

        Action=[None for i in range((4*N-1)*N*3)]
        self._inverse_action={}
        for i in range(N) :
            for j in range(4*N-1) :
                s1='X_'+str(i).zfill(2)+'_'+str(j+1).zfill(2)
                s2='X_'+str(i).zfill(2)+'_'+str(4*N-(j+1)).zfill(2)
                k=j+(4*N-1)*i
                Action[k]=Action_Type(actidx=k,actstr=s1)
                self._inverse_action[s1]=s2
        for i in range(N) :
            for j in range(4*N-1) :
                s1='Y_'+str(i).zfill(2)+'_'+str(j+1).zfill(2)
                s2='Y_'+str(i).zfill(2)+'_'+str(4*N-(j+1)).zfill(2)
                k=j+(4*N-1)*i+((4*N-1)*N)
                Action[k]=Action_Type(actidx=k,actstr=s1)
                self._inverse_action[s1]=s2
        for i in range(N) :
            for j in range(4*N-1) :
                s1='Z_'+str(i).zfill(2)+'_'+str(j+1).zfill(2)
                s2='Z_'+str(i).zfill(2)+'_'+str(4*N-(j+1)).zfill(2)
                k=j+(4*N-1)*i+((4*N-1)*N)*2
                Action[k]=Action_Type(actidx=k,actstr=s1)
                self._inverse_action[s1]=s2

        self.action_enum=Action

        self.encoded_shape = (6, N*N*6)

        # colors (for visualization)
        self.sq_colors = ['Y','B','R','G','O','W']

        self.razv_idx_list=[
            [
            [-1,-1, 0, 2,-1,-1,-1,-1],
            [-1,-1, 1, 3,-1,-1,-1,-1],
            [ 6, 7,10,11,15,14,19,18],
            [ 4, 5, 8, 9,13,12,17,16],
            [-1,-1,21,23,-1,-1,-1,-1],
            [-1,-1,20,22,-1,-1,-1,-1]
            ],
            [
            [-1,-1,-1, 0, 3, 6,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1, 1, 4, 7,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1, 2, 5, 8,-1,-1,-1,-1,-1,-1],
            [15,16,17,24,25,26,35,34,33,44,43,42],
            [12,13,14,21,22,23,32,31,30,41,40,39],
            [ 9,10,11,18,19,20,29,28,27,38,37,36],
            [-1,-1,-1,47,50,53,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,46,49,52,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,45,48,51,-1,-1,-1,-1,-1,-1],
            ]
        ]

        self.razv_idx=self.razv_idx_list[self.N-2]

    # wrapper functions
    def is_goal(self, state):
        assert isinstance(state, State_type)
        # return self.state_cost(state)==self.goal_cost
        a=state.sq_pos
        for i in range(6) :
            b=a[i*self.N*self.N:(i+1)*self.N*self.N]
            if not np.all(np.array(b)==b[0]) : return False
        return True

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
        a=_permute(state.sq_pos,self.cube_move_set[act_num])

        return State_type(sq_pos=tuple(a))

    def inverse_action(self, action):
        assert isinstance(action, Action_Type)
        return self._inverse_action[action]

    def render(self, state):
        assert isinstance(state, State_type)

        a=state.sq_pos
        r=copy.deepcopy(self.razv_idx)
        ridx=np.asarray(self.razv_idx)

        for i in range(len(a)) :
            idxm = np.argwhere(ridx==i)
            r[idxm[0][0]][idxm[0][1]]=self.sq_colors[a[i]]

        idxm = np.argwhere(ridx==-1)
        for i in range(len(idxm)) :
            r[idxm[i][0]][idxm[i][1]]=' '

        s=''
        for i in range(self.N*3) : s=s+''.join(r[i])+'\n'

        return s

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
        M=4*self.N-1
        if int(mov1[0]/M)==int(mov2[0]/M) :
            return 1
        return 0

    def sample_action(self, prev_action=None):
        while True:
            res = self.action_enum[random.randrange(len(self.action_enum))]
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

    # def scramble_cube(self, scrambles_count, return_inverse=False, include_initial=False):
    #     """
    #     Generate sequence of random square scrambles
    #     :param scrambles_count: count of scrambles to perform
    #     :param return_inverse: if True, inverse action is returned
    #     :return: list of tuples (depth, state[, inverse_action])
    #     """
    #     assert isinstance(scrambles_count, int)
    #     assert scrambles_count > 0

    #     state = self.initial_state
    #     result = []
    #     if include_initial:
    #         assert not return_inverse
    #         result.append((1, state))
    #     prev_action = None
    #     for depth in range(scrambles_count):
    #         action = self.sample_action(prev_action=prev_action)
    #         state = self.transform(state, action)
    #         prev_action = action
    #         if return_inverse:
    #             inv_action = self.inverse_action(action)
    #             res = (depth+1, state, inv_action)
    #         else:
    #             res = (depth+1, state)
    #         result.append(res)
    #     return result

    def scramble_cube(self, scrambles_count, initial_state=None, return_inverse=False, include_initial=False):
        """
        Generate sequence of random square scrambles
        :param scrambles_count: count of scrambles to perform
        :param return_inverse: if True, inverse action is returned
        :return: list of tuples (depth, state[, inverse_action])
        """
        assert isinstance(scrambles_count, int)
        assert scrambles_count > 0

        if initial_state==None :
            state = self.initial_state
        else :
            state = initial_state
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

    # def state_cost(self, state):
    #     """
    #     Estimates cost of state from viewpoint of squares mutual arrangement
    #     """
    #     N=self.N

    #     def GetHorizontalAlingment(a) :
    #         r=[i for i, j in zip(a, a[1:]+ (a[0],)) if i == j]
    #         return len(r)

    #     assert isinstance(state, State_type)
    #     a=state.sq_pos
    #     s=0
    #     for i in range(N) :
    #         s=s+GetHorizontalAlingment(a[i*N:(i+1)*N])
    #     return s

    def ConvertToState(self,a):
        return State_type(sq_pos=tuple(a))
