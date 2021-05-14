## gen_all_pos.py
import numpy as np

def gen_all_pos(N) :
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
        pos=-np.ones((n),dtype=int)

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

    q=[]
    for i in range(numstates) :
        a=idx2pos(i,np.arange(N,dtype=int),np.ones((N,),dtype=int)*N,N*N)
        q.append(a)

    return q