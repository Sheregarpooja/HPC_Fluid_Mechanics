import numpy as np
import matplotlib.pyplot as plt

# Define the D2Q9 lattice velocity vectors and weights
c = np.array([[ 0, 1, 0,-1, 0, 1,-1,-1, 1],[ 0, 0, 1, 0,-1, 1, 1,-1,-1]]).T
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]).T
#  the fluid density at each lattice node
def density(f):
    a=np.sum(f, axis=0) 
    return a

#  the fluid velocity at each lattice node
def velcoity(f,d):
    return np.dot(f.T,c).T/d

# Collision step: Adjusts distribution functions towards their equilibrium
def collision(f,v,d,relaxation=0.5):
    eq = equilibrium(d,v)
    f -= relaxation * (f-eq)    
    return f

#  equilibrium distribution function for each direction
def equilibrium(d,v):
    vel_mag = v[0,:,:] ** 2 + v[1,:,:] ** 2
    temp = np.dot(v.T,c.T).T
    sq_velocity = temp ** 2
    f_eq = (((1 + 3*(temp) + 9/2*(sq_velocity) - 3/2*(vel_mag)) * d ).T * w).T
    return f_eq
# Streaming step: Shift distribution functions based on velocity vectors
def streaming(f):
    for i in range(9):
        f[i,:,:] = np.roll(f[i,:,:],c[i], axis = (0,1))
    return f


#boundary condition for lid driven 
def boundry_condition(f,lid_vel):
    # Top and bottom no-slip boundaries (moving lid at the top)
    f[2, :, 1] = f[4, :, 0] 
    f[5, :, 1] = f[7, :, 0]
    f[6, :, 1] = f[8, :, 0]
    f[4, :, -2] = f[2, :, -1]
    f[7, :, -2] = f[5, :, -1] - 1 / 6 * lid_vel
    f[8, :, -2] = f[6, :, -1] + 1 / 6 * lid_vel
    f[[3,6,7],-2,:]=f[[1,8,5],-1,:]
    f[[1,5,8],1,:]=f[[3,7,6],0,:]
    return f


def plotter(step,length,v,relaxation):
        plt.clf()
        x = np.arange(0, length-2)
        y = np.arange(0, length-2)
        X, Y = np.meshgrid(x, y)
        speed = np.sqrt(v[0,2:-2,2:-2].T ** 2 + v[1,2:-2,2:-2].T ** 2)
        plt.streamplot(X,Y,v[0,2:-2,2:-2].T,v[1,2:-2,2:-2].T, color=speed, cmap= plt.cm.jet)
        ax = plt.gca()
        ax.set_xlim([0, length+1])
        ax.set_ylim([0, length+1])
        titleString = "Sliding Lid (Gridsize " + "{}".format(length) + "x" + "{}".format(length)
        titleString += ",  $\\omega$ = {:.2f}".format(relaxation) + ", steps = {}".format(step) + ")"
        plt.title(titleString)
        plt.xlabel("x-Position")
        plt.ylabel("y-Position")
        fig = plt.colorbar()
        fig.set_label("Velocity u(x,y,t)", rotation=270,labelpad = 15)
        savestring="slidinglid_step_"+str(step)+".png"
        plt.savefig(savestring)

raynolads=1000
length=300
lid_vel=0.1
relaxation=((2 * raynolads) / (6 * length * lid_vel + raynolads))
steps=10000
rate=100

den=np.ones((length+2,length+2))
vel=np.zeros((2,length+2,length+2))
f=equilibrium(den,vel)
for step in range(steps):
    f=streaming(f)
    f=boundry_condition(f,lid_vel)
    den=density(f)
    vel=velcoity(f,den)
    f=collision(f,vel,den,relaxation)
    if (step % rate) == 0 and step > 0:
        rate = rate*2
        plotter(step,length,vel,relaxation)

plotter(steps,length,vel,relaxation)
plt.show()
