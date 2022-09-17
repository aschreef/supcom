import numpy as np 
import math 
from numpy import linalg as LA 
import matplotlib.pyplot as plt


try :
    n=int(input("donner la dimension la matrice"))
    L=int(input("danner la longuer de la barre"))
    T0=float(input("donner la température à l'origine"))
    Tl=float(input("donner la tepérature à l'extrimité"))
except :
    n=100
    L=100
    T0=30.0
    Tl=50.0

def matriceEqChaleur(n):

    A=np.zeros((n-1,n-1))
    A[0,0]= 2
    A[0,1]=-1
    A[1,0]=-1
    A[n-2,n-2]=2
    A[n-2,n-3]=-1
    A[n-3,n-2]=-1

    for i in range(1,n-2):
        A[i,i]=2
        A[i,i+1]=-1
        A[i,i-1]=-1

    return(A)


def fonctionSource(n,T0,Tl,L):
    B=np.zeros(n-1)
    h=L/n
    B[0]   =math.sin(h) + T0/h**2
    B[n-2] =math.sin(L-h) + Tl/h**2
    for i in range(1,n-2):
        B[i]=math.sin((i+1)*h)
    return(B)


def defSystem(n,T0,Tl,L):
    A=matriceEqChaleur(n)
    B=fonctionSource(n,T0,Tl,L)
    X=np.ones(n-1)
    return(A,B,X)


def gradientPasFixe(A,B,X0,pas,tolerence,max_iteration):
    compteur=0
    X=X0
    while LA.norm(np.matmul(A,X)-B)>tolerence:
        X=X-pas * (np.matmul(A,X)-B) 
        compteur +=1
        if compteur > max_iteration :
            print("l'algorithme a divergé")
            break
    return(X,compteur)


def pasFixeOptimal(A,B,X0,max_iteration,bi,bs,ech,tolerence=0.001):
    p = np.linspace(bi,bs,ech)
    iteration=[]
    for i in p :
        x_opt,cpt=gradientPasFixe(A,B,X0,i,tolerence,max_iteration)
        iteration.append(cpt)
    plt.plot(p,iteration)
    plt.show()
    r = iteration.index(min(iteration))
    return(p[r])

def gradientPasOptimale(A,B,X0,tolerence,maxIterations):
    cpt = 0
    X=X0
    R = LA.norm((np.matmul(A,X)-B))
    P = np.matmul(A,X)-B
    while R > tolerence :
        pas = np.dot(P,P)/np.dot(np.matmul(A,P),P)
        X = X - pas * P
        cpt += 1
        P = np.matmul(A,X)-B
        R = LA.norm((np.matmul(A,X)-B))
        if maxIterations < cpt :
            print("le model diverge")
            break
    return X,cpt

def gradientPasConjugue(A,B,X,tolerence,max_iterations):
    G=np.matmul(A,X)-B
    d=np.multiply(G, -1)
    R=LA.norm(np.matmul(A,X)-B)
    pas =  np.dot(d,d)/np.dot(np.matmul(A,d),d)
    compteur=0
    X= X + np.multiply(d,pas)
    while R > tolerence :
        G=np.matmul(A,X)-B
        beta = np.dot(np.matmul(A,G),d)/np.dot(np.matmul(A,d),d)
        d= beta*d -1* G
        pas = -1*(np.dot(d,G)/np.dot(np.matmul(A,d),d))
        X=X+pas*d
        R=LA.norm((np.matmul(A,X)-B))
        compteur += 1
        if compteur > max_iterations :
            print("le model a divergé")
            break 
    return(X,compteur)




def modelComplet():
    
    A,B,X=defSystem(n,T0,Tl,L)
    print("------------ test du model à pas fixe ----------------------")
    print("------------- calul du meilleure pas ------------------------")
    pasOpt=pasFixeOptimal(A,B,X,50000,0.05,1,50,0.08)
    print("la meilleur pas est : ",pasOpt)
    print("------------ test du model à pas optimale ----------------------")
    X_opt,cpt=gradientPasOptimale(A,B,X,0.1,20000)
    print("solution avec pas optimale: ", X_opt)
    print("convergence après ",cpt , " pas")
    print("------------ test du model à pas conjugué ----------------------")
    Xo,c=gradientPasConjugue(A,B,X,0.001,20000)
    print("la solution avec le model à pas conjugué est : ",Xo)
    print("avec un nombre d'itérations : ",c)

    try :
        min=int(input("donner le minimum de la valeur de N"))
        max=int(input("donner le maximum de la valeur de N"))
    except : 
        min=10
        max=100

    k=max-min+1
    N_range = [0]*k
    for i in range(k):
        N_range[i]=k+i
    Num_iterations1=[]
    Num_iterations2=[]
    for i in N_range:
        A,B,X=defSystem(i,T0,Tl,L)
        X_opt1,it1=gradientPasOptimale(A,B,X,0.08,20000)
        X_opt2,it2=gradientPasConjugue(A,B,X,0.08,1000)
        Num_iterations1.append(it1)
        Num_iterations2.append(it2)
    Num_iterations1=np.array(Num_iterations1)
    Num_iterations2=np.array(Num_iterations2)
    N_range = np.array(N_range)
    plt.plot(N_range,Num_iterations1)
    plt.show()
    plt.plot(N_range,Num_iterations2)
    plt.show()



modelComplet()



