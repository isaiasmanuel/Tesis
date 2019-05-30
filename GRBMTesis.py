# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:38:04 2019

@author: kille
"""
import mpmath as mpm
import numpy as np
import time 
import torchvision
import torchvision.transforms as transforms
from scipy import stats



#################

def E(V,H,b,c,W,s2):
    return(np.linalg.norm(V-b)**2/(2*s2)-(np.transpose(c)@H)[0]-(np.transpose(V)@W@H)[0]/s2)

def PH_X(V,j,c,s2,W):
    return(1/(1+mpm.exp((-c[j]-np.transpose(V)@(W[:,j])/s2)[0])))


def PX_H(x,H,i,b,s2,W):
    return(mpm.npdf(x,mu=b[i][0]+np.transpose(W[i,:])@H,sigma=np.sqrt(s2)))
    
def PXH(V,H,b,c,W,s2):    #Salvo constante
    return mpm.exp(-E(V,H,b,c,W,s2))    

def H1(V,c,s2,W):
    Vector=np.reshape(np.ones(Oc),(Oc,1))
    for i in range(Oc):
        Vector[i]=PH_X(V,i,c,s2,W)
    return(Vector)

def Parb(V,b,s2):
    #mpm.matrix((V-b)/s2)
#    X=mpm.matrix((V-b)/s2)
    X=((V-b)/s2)
#    return(np.array(X.tolist(), dtype=float))
    return(np.array(X, dtype=float))

def Pars(V,b,s2,c,W):
    return(((np.linalg.norm(V-b)**2-2*(np.transpose(V)@W@H1(V,c,s2,W))[0])/(s2**(3/2)))[0])

# =============================================================================
# 
# 
# def Gibs(Samp,Step,V,H,b,c,W,s2):
#     Vmuestra=np.ones((28*28,Samp,1))
#     Vaux=np.copy(V)
#     for k in range(Samp):
#         for j in range(Step):
#             Uniformes=stats.uniform().rvs(Oc)
#             for i in range(Oc):
#                 if(PH_X(Vaux,i,c,s2,W)>Uniformes[i]):
#                     H[i]=1        
#                 else:
#                     H[i]=0
#             Normales=stats.norm(loc=b[i][0]+np.transpose(W[i,:])@H,scale=s2).rvs(28*28)
#         
#             for i in range(28*28):
#                 if(Normales[i]>1):
#                     Vaux[i]=1        
#                 elif(Normales[i]<0):
#                     Vaux[i]=0
#                 else:
#                     Vaux[i]=Normales[i]
#         Vmuestra[:,k]=Vaux
#     return(Vmuestra)
# 
# =============================================================================
k=0
i=0
j=0
# =============================================================================
# def Gibs(Step,Muestra,H,b,c,W,s2):    
#     Samp=Muestra.shape[1]
#     Vmuestra=np.ones((28*28,Samp,1))
#     for k in range(Samp):    
#         Vaux=np.copy(Muestra[:,k])
#         for j in range(Step):
#             Uniformes=stats.uniform().rvs(Oc)
#             for i in range(Oc):
#                 if(PH_X(Vaux,i,c,s2,W)>Uniformes[i]):
#                     H[i]=1        
#                 else:
#                     H[i]=0
#             Normales=stats.norm(loc=b[i][0]+np.transpose(W[i,:])@H,scale=s2).rvs(28*28)
#         
#             for i in range(28*28):
#                 if(Normales[i]>1):
#                     Vaux[i]=1        
#                 elif(Normales[i]<0):
#                     Vaux[i]=0
#                 else:
#                     Vaux[i]=Normales[i]
#         Vmuestra[:,k]=Vaux
#     return(Vmuestra)
# 
# =============================================================================
def Gibs(Step,Muestra,H,b,c,W,s2):    
    Samp=Muestra.shape[1]
    Vmuestra=np.ones((28*28,Samp,1))
    for k in range(Samp):    
        Vaux=np.copy(Muestra[:,k])
        for j in range(Step):
            Uniformes=stats.uniform().rvs(Oc)
            for i in range(Oc):
                if(PH_X(Vaux,i,c,s2,W)>Uniformes[i]):
                    H[i]=1        
                else:
                    H[i]=0                    
            for i in range(28*28):
                    Vaux[i]=stats.norm(loc=b[i][0]+np.transpose(W[i,:])@H,scale=np.sqrt(s2)).rvs(1)[0]
        Vmuestra[:,k]=Vaux
    return(Vmuestra)


def EParb(Muestra,b,s2):
    suma=0
    for i in range(Muestra.shape[1]):
        suma+=Parb(Muestra[:,i],b,s2)
    return(suma/Muestra.shape[1])

def EParc(Muestra,c,s2,W):
    suma=0
    for i in range(Muestra.shape[1]):
        suma+=H1(Muestra[:,i],c,s2,W)
    return(suma/Muestra.shape[1])

def EParw(Muestra,c,s2,W):
    suma=0
    for i in range(Muestra.shape[1]):
        suma+=  Muestra[:,i]@np.transpose(H1(Muestra[:,i],c,s2,W))/s2
    return(suma/Muestra.shape[1])

def EPars(Muestra,c,s2,W,b):
    suma=0
    for i in range(Muestra.shape[1]):
        suma+=  Pars(Muestra[:,i],b,s2,c,W)
    return(suma/Muestra.shape[1])

def logPX(V,b,s2,Oc,W,c):
    dens=1
    for i in range(28*28):
        dens=dens*mpm.npdf(V[i,0],mu=b[i,0],sigma=np.sqrt(Oc*s2))
    Norm1=dens
    const=mpm.sqrt(2*(np.pi)*Oc*s2)**(28*28)    
    Densfinal=1
    for j in range(Oc):
        dens=1
        for i in range(28*28):
            dens=dens*mpm.npdf(V[i,0],mu=b[i,0]+Oc*W[i,j],sigma=np.sqrt(Oc*s2))
        Norm2=dens
        Shift=mpm.exp(((mpm.norm(b[:,0]+Oc*W[:,j])**2-mpm.norm(b[:,0])**2)/(2*Oc*s2)+c[j])[0])
        Densfinal=const*Densfinal*(Norm1+Shift*Norm2)
    return(mpm.log(Densfinal))

def logveros(Muestra,b,s2,Oc,W,c):
    suma=0
    for i in range(Muestra.shape[1]):
        suma+=logPX(Muestra[:,i],b,s2,Oc,W,c)
    return(suma)

######################Inicializar
Oc=3*3 #Numero de unidades ocultas
SampleSize=50000

#transform = transforms.Compose([transforms.ToTensor()])
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)


#V=np.reshape(trainset[0][0].detach().numpy(),(28*28,1))


Muestra=np.ones((28*28,SampleSize,1))
for i in range(SampleSize):
    Muestra[:,i]=np.reshape(trainset[i][0].detach().numpy(),(28*28,1))


H=np.reshape(np.random.randint(0,2,Oc),(Oc,1))


#####Inicializacion
#b=np.reshape(stats.uniform().rvs(28*28),(28*28,1))
#c=np.reshape(stats.uniform().rvs(Oc),(Oc,1))
#W=np.reshape(np.ones(28*28*Oc),(28*28,Oc))-(1-10**(-10))

b=np.reshape(np.ones(28*28),(28*28,1))-1
c=np.reshape(stats.uniform().rvs(Oc),(Oc,1))/10
W=np.reshape(np.ones(28*28*Oc),(28*28,Oc))-(1-10**(-10))






##################### Inicial para b

# =============================================================================
# suma=0
# for i in range(Muestra.shape[1]):
#     suma+=Muestra[:,i]    

# suma/SampleSize
# 
# b=suma/SampleSize
# 
# =============================================================================

#####Inicializacion
# =============================================================================
# from numpy import genfromtxt
# b= np.reshape(genfromtxt('C:/Users/kille/Desktop/b.csv', delimiter=','),(28*28,1))
# c= np.reshape(genfromtxt('C:/Users/kille/Desktop/c.csv', delimiter=','),(Oc,1))
# W= np.reshape(genfromtxt('C:/Users/kille/Desktop/W.csv', delimiter=','),(28*28,Oc))
# 
# 
# =============================================================================

######Epsilon
e=.1
Step=1
batch=10
s2=.1


#############
Inicio=time.time()
for i in range(10000):
    B=np.random.randint(0, high=Muestra.shape[1], size=batch, dtype='l')
    GibSample=Gibs(Step,Muestra[:,B,:],H,b,c,W,s2)
    Nb=b+e*(EParb(Muestra[:,B,:],b,s2)-EParb(GibSample,b,s2))
    Nc=c+e*(EParc(Muestra[:,B,:],c,s2,W)-EParc(GibSample,c,s2,W))
    Nw=W+e*(EParw(Muestra[:,B,:],c,s2,W)-EParw(GibSample,c,s2,W))
#    Ns=np.sqrt(s2)+e*(EPars(Muestra,c,s2,W,b)-EPars(GibSample,c,s2,W,b))
    b=Nb
    c=Nc
    W=Nw
#    print(s2,Ns)
#    if Ns>0:
#    s2=Ns**2
    print(i)    
#    print(i)
#    if i%1==0:
#        print("Veros",logveros(Muestra,b,s2,Oc,W,c))
Final=time.time()


#np.savetxt("W.csv", W, delimiter=",")
#np.savetxt("b.csv", b, delimiter=",")
#np.savetxt("c.csv", c, delimiter=",")

Final-Inicio    


#logveros(Muestra,b,0.00001,Oc,W,c)


###############
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    


V0=np.reshape(Muestra[:,0],(784,1,1))
X=GibSample
M=1
imshow(torchvision.utils.make_grid(torch.from_numpy(np.reshape(2*np.arctan(X[:,M])/np.pi,(28,28)))))
imshow(torchvision.utils.make_grid(torch.from_numpy(np.reshape(X[:,M],(28,28)))))
imshow(torchvision.utils.make_grid(torch.from_numpy(np.reshape(Muestra[:,M,:],(28,28)))))



#Ver que de algo uniforme
o=0
X2=(X[:,o]-min(X[:,o]))/(max(X[:,o])-min(X[:,o]))
imshow(torchvision.utils.make_grid(torch.from_numpy(np.reshape(X2,(28,28)))))




logPX(Muestra[:,0,:],b,s2,Oc,W,c)

logPX(Muestra[:,0,:]-Muestra[:,0,:]+10,b,s2,Oc,W,c)

logPX(Muestra[:,0,:]-Muestra[:,0,:]+1,b,s2,Oc,W,c)

Muestra[:,0,:].shape

##############
logPX(Muestra[:,0,:]-Muestra[:,0,:]+np.transpose(znp),b,s2,Oc,W,c)
logPX(Muestra[:,0,:]-Muestra[:,0,:]+np.transpose(Nnp),b,s2,Oc,W,c)
np.transpose(znp)
V0-V0


l=0
alfa=0+10**(0) #regularizador
beta=.001 #gradiente
delta=1
C=2
#for l in range(100):
while F.softmax(net(N),1)[0][C]<0.99999:
    if l==0 :
#            z=images[:1]
#            z=images[:1]-images[:1]+torch.from_numpy(np.float32(samples[29:,:]))            
        z=images[:1]-images[:1]+torch.ones(28,28,requires_grad=True)+torch.empty(28, 28).uniform_(-2, 0)                 
#                z=images[:1]-images[:1]+torch.from_numpy(SevenSegment( seg )/256).float()
#            z=images[:1]-images[:1]+torch.from_numpy(samples[29:,:]).float()
#        z.requires_grad=True
            
#            z.requires_grad=True
#            z=torch.ones(28,28,requires_grad=True)+torch.empty(28, 28).uniform_(-2, 0)    #-.99
        #        net.zero_grad()            
    znp=np.reshape(z.detach().numpy(),(784,1))

    suma=0
    dens=1
    for i in range(28*28):
        dens=dens*mpm.npdf(np.float(znp[i,0]),mu=b[i,0],sigma=np.sqrt(Oc*s2))
    Norm1=dens

    for j in range(Oc):
        dens=1
        for i in range(28*28):
            dens=dens*mpm.npdf(np.float(znp[i,0]),mu=b[i,0]+Oc*W[i,j],sigma=np.sqrt(Oc*s2))
        Norm2=dens
        Shift=mpm.exp(((mpm.norm(b[:,0]+Oc*W[:,j])**2-mpm.norm(b[:,0])**2)/(2*Oc*s2)+c[j])[0])
        suma+=(W[:,j]*Norm2*Shift)/(Norm1+Shift*Norm2)        
    suma=np.reshape(np.array(suma.tolist(),dtype=np.float64),(784,1))


    R=torch.autograd.grad(torch.log(F.softmax(net(z),1)[0][C]), z, retain_graph=True)[0]-alfa*torch.from_numpy(np.reshape(((znp-b)-suma),(28,28))).float()
#    W=delta*torch.autograd.grad(torch.log(F.softmax(net(z),1)[0][C]), z, retain_graph=True)[0]-alfa*torch.from_numpy(np.reshape((gra+suma),(28,28))).float()
    N=z+beta*R    
#    N[N>1]=1
#            N[N<0]=0       
#    N[N<-1]=-1   
    Nnp=np.reshape(N.detach().numpy(),(1,784))
    print(delta*np.log(F.softmax(net(N),1)[0][C].detach().numpy())+ alfa*logPX(Muestra[:,0,:]-Muestra[:,0,:]+np.transpose(Nnp),b,s2,Oc,W,c),l,F.softmax(net(N),1)[0][C])
    z=torch.autograd.Variable(N.data,requires_grad=True)
#    print(l,F.softmax(net(N),1)[0][C])
    l+=1


F.softmax(net(N),1)[0][C].detach().numpy()
imshow(torchvision.utils.make_grid(z.detach()))

imshow(torchvision.utils.make_grid(N.detach()))

X=(N-torch.min(N))/(torch.max(N)-torch.min(N))
F.softmax(net(X),1)[0][C].detach().numpy()
imshow(torchvision.utils.make_grid(X.detach()))


