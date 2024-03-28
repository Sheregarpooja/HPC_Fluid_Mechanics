import numpy as np

x=15
y=10
v=9
c = np.array([[ 0, 1, 0,-1, 0, 1,-1,-1, 1],[ 0, 0, 1, 0,-1, 1, 1,-1,-1]]).T
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]).T
f=np.random.rand(v,x,y)
for k in range(v):
    for i in range(x):
        for j in range(y):
            f[k,i,j]=np.random.randint(2)

         
            
def density(f):
    a=np.sum(f, axis=0) #density 
   # print("sum")
    print(a.shape)
    return a

def velcoity(f,d):
    print((np.dot(f.T,c).T/d).shape)
    return np.dot(f.T,c).T/d

def collision(f,v,d,relaxation=0.5):
    eq = equilibrium(d,v)
    f -= relaxation * (f-eq)    
    return f

def equilibrium(d,v):
    vel_mag = v[0,:,:] ** 2 + v[1,:,:] ** 2
    print(vel_mag.shape)
    temp = np.dot(v.T,c.T).T
    print(temp.shape)
    sq_velocity = temp ** 2
    f_eq = (((1 + 3*(temp) + 9/2*(sq_velocity) - 3/2*(vel_mag)) * d ).T * w).T
    return f_eq

def streaming(f):
    for i in range(9):
        f[i,:,:] = np.roll(f[i,:,:],c[i], axis = (0,1))
    return f

den=density(f)
vel=velcoity(f,den)

steps=10
for step in range(steps):
    den=density(f)
    vel=velcoity(f,den)
    f=collision(f,vel,den)
    f=streaming(f)

print(w.shape)    