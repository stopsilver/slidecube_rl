# SlideSquare_DP_statevalue.py

import numpy as np
from slidesquare_env import SquareEnv

# slidesquare_type='square4x4'   # in reality "2x2"

# square_env=slidesquares.get(slidesquare_type)

square_env=SquareEnv(3)

## generate list of all states
N=square_env.encoded_shape[0]

numstates=np.int(np.math.factorial(N*N)/np.power(np.math.factorial(N),N))

def pos2idx_init() :
    a=np.zeros((N*N,N*N))
	# fill Pascal's triangle
    a[0][0] = 1
    a[0][1] = 0
    for i in range(1,N*N) :
        for j in range(i+1) :
            ul = 0
            u = a[i - 1][j]
            if j>0 : ul= a[i - 1][j-1]
            a[i][j] = ul+u
        if i < N*N - 1 : a[i][i + 1] = 0
    return a


PascalTriangle=pos2idx_init()

def nchoosek_deenum(n,k,e) :
    idx=[i for i in range(k)]

    if e==0 :
        return idx
    n1=n-1
    while e>0 :
        while e< PascalTriangle[n1][k] : n1-=1
        idx[k-1]=n1
        e-=PascalTriangle[n1][k]
        n1-=1
        k-=1
    if k>0 :
        for i in range(k) : idx[i]=i
    return idx

def idx2pos(enum_idx,tile_list,tile_num,n) :
    pos=-np.ones((n))

    for i in range(len(tile_list)-1,-1,-1) :
        # get partial enumeration
        tn1=0
        if (i-1)>=0 : tn1=tile_num[i-1]
        d=PascalTriangle[n-tile_num[i]][tn1]
        e=enum_idx//d
        ind=nchoosek_deenum(n,tile_num[i],e)
        clr=tile_list[i]
        m=0
        cnt=0
        for j in range(N*N) :
            if pos[j]<0 :
                if cnt==ind[m] :
                    pos[j]=clr
                    m+=1
                cnt+=1
                if m>=N : break
        enum_idx-=e*d
        n-=tile_num[i]

    idx=np.where(pos<0)
    pos[idx]=N-1

    return pos

states={}
idx_states={}
for i in range(numstates) :
    a=idx2pos(i,[0, 1],[N, N],N*N)
    rn=square_env.ConvertToState(a.astype(int))
    states[i]=rn
    idx_states[rn]=i

## render all states
# for key in states.keys() :
#     print(square_env.render(states[key]))

## interesting idea from (https://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary)
# mydict = {'george': 16, 'amber': 19}
# print(list(mydict.keys())[list(mydict.values()).index(16)])  # Prints george

def GetNewStateAndReward(idx,Move) :
    # move to new state
    st=states[idx]
    st1=square_env.transform(st,square_env.action_enum[Move])
    # reward
    cost=square_env.state_cost(st)
    # cost1=square_env.state_cost(st1)
    # R=cost1-cost
    R=cost
    R=max(R,0)
    return idx_states[st1], R

V=np.zeros((numstates))

gamma=0.9

for k in range(100) :            # iteration
    Vnew=np.zeros((numstates))
    for sidx in range(numstates) :         # for all states
        s=0
        for j in range(len(square_env.action_enum)) :
            sidx1,R=GetNewStateAndReward(sidx,j)
            SV1=V[sidx1]
            s=s+(R+gamma*SV1)

        s=s/len(square_env.action_enum)
        Vnew[sidx]=s

    V=Vnew

    print('k='+str(k+1))
    print(V[0:10])
