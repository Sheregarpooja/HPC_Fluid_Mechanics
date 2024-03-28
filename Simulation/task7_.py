import numpy as np
#import matplotlib.pyplot as plt
from mpi4py import MPI
import time
import sys

#record start of simulation
start = time.time()

# Define the D2Q9 lattice velocity vectors and weights
c = np.array([[ 0, 1, 0,-1, 0, 1,-1,-1, 1],[ 0, 0, 1, 0,-1, 1, 1,-1,-1]]).T
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]).T

# Calculate fluid density at each lattice node by summing over all directions
def density(f):
    a=np.sum(f, axis=0) 
    return a
# Calculate fluid velocity at each lattice node
def velcoity(f,d):
    return np.dot(f.T,c).T/d

# Collision step: Adjusts distribution functions towards equilibrium
def collision(f,v,d,relaxation=0.5):
    eq = equilibrium(d,v)
    f -= relaxation * (f-eq)    
    return f

# Compute equilibrium distribution function for each direction
def equilibrium(d,v):
    vel_mag = v[0,:,:] ** 2 + v[1,:,:] ** 2
    temp = np.dot(v.T,c.T).T
    sq_velocity = temp ** 2
    f_eq = (((1 + 3*(temp) + 9/2*(sq_velocity) - 3/2*(vel_mag)) * d ).T * w).T
    return f_eq

# Streaming step: Shifts distribution functions according to their velocities
def streaming(f):
    for i in range(9):
        f[i,:,:] = np.roll(f[i,:,:],c[i], axis = (0,1))
    return f

# Apply boundary conditions for lid-driven cavity flow
def boundry_condition(f,lid_vel,b,t,r,l):
    if b:
        f[2, :, 1] = f[4, :, 0] 
        f[5, :, 1] = f[7, :, 0]
        f[6, :, 1] = f[8, :, 0]
    if t:
        f[4, :, -2] = f[2, :, -1]
        f[7, :, -2] = f[5, :, -1] - 1 / 6 * lid_vel
        f[8, :, -2] = f[6, :, -1] + 1 / 6 * lid_vel
    if r:
        f[[3,6,7],-2,:]=f[[1,8,5],-1,:]
    if l:
        f[[1,5,8],1,:]=f[[3,7,6],0,:]

    return f

# Function for MPI communication between adjacent subdomains
def communicate(mpi,f,br,bl,bb,bt,nr,nl,nb,nt):
     
    # Communicate distribution functions across subdomain boundaries
    # 'br', 'bl', 'bb', 'bt' indicate whether the current subdomain has boundary right, left, bottom, and top
    # 'nr', 'nl', 'nb', 'nt' are the right, left, bottom, and top neighbors respectively
    # This function performs non-blocking sends and receives to update ghost layers
      
        if not br:
            recvbuf = f[:, -1, :].copy()
            mpi.Sendrecv(f[:,-2, :].copy(), nr, recvbuf=recvbuf, sendtag = 10, recvtag = 20)
            f[:, -1, :] = recvbuf
        if not bl:
            recvbuf = f[:, 0, :].copy()
            mpi.Sendrecv(f[:, 1, :].copy(), nl, recvbuf=recvbuf, sendtag = 20, recvtag = 10)
            f[:, 0, :] = recvbuf
        if not bb:
            recvbuf = f[:,: ,0 ].copy()
            mpi.Sendrecv(f[:, :,1 ].copy(), nb, recvbuf=recvbuf, sendtag = 30, recvtag = 40)
            f[:, :, 0] = recvbuf
        if not bt:
            recvbuf = f[:, :, -1].copy()
            mpi.Sendrecv(f[:, :, -2].copy(), nt, recvbuf=recvbuf, sendtag = 40, recvtag = 30)
            f[:, :, -1] = recvbuf
        
        return f

def collapse(f):
        full_grid = np.ones((9, length, length))
        if rank == 0:
            full_grid[:,0:x-2,0:y-2] = f[:,1:-1,1:-1]
            temp = np.zeros((9,x-2,y-2))
            for i in range(1,size_comm):
                communicator.Recv(temp,source = i,tag = i)
                x_p,y_p = i % int(np.sqrt(size_comm)),i // int(np.sqrt(size_comm))
                full_grid[:,(0 + (x-2)*x_p):((x-2) + (x-2)*x_p),(0 + (y-2)*y_p):((y-2) + (y-2)*y_p)] = temp
        else:
            communicator.Send(f[:,1:-1,1:-1].copy(),dest=0, tag = rank)

        return full_grid


def plotter(step,length,v,f,relaxation):
        if rank==0:
            file="f_mpipslidinglid_proc_"+str(size_comm)+"_grid_"+str(length)+"_steps_"+str(steps)+".npy"
            np.save(file,f)
            #plt.clf()
            #x = np.arange(0, length)
            #y = np.arange(0, length)
            #X, Y = np.meshgrid(x, y)
            #speed = np.sqrt(v[0].T ** 2 + v[1].T ** 2)
            #plt.streamplot(X,Y,v[0].T,v[1].T, color=speed, cmap= plt.cm.jet)
            #ax = plt.gca()
            #ax.set_xlim([0, length+1])
            #ax.set_ylim([0, length+1])
            #titleString = "Sliding Lid (Gridsize " + "{}".format(length) + "x" + "{}".format(length)
            #titleString += ",  $\\omega$ = {:.2f}".format(relaxation) + ", steps = {}".format(step) + ")"
            #plt.title(titleString)
            #plt.xlabel("x-Position")
            #plt.ylabel("y-Position")
            #fig = plt.colorbar()
            #fig.set_label("Velocity u(x,y,t)", rotation=270,labelpad = 15)
            #savestring="slidinglidmpi_step_"+str(step)+".png"
            #plt.savefig(savestring)
            txt_flie="f_mpipslidinglid_proc_"+str(size_comm)+"_grid_"+str(length)+"_steps_"+str(steps)+".txt"
            ftxt = open(txt_flie,"w")
            ftxt.write(str(time.time() - start))
            ftxt.close()


raynolads=1000
length=int(sys.argv[1])
lid_vel=0.1
relaxation=((2 * raynolads) / (6 * length * lid_vel + raynolads))
steps=int(sys.argv[2])
rate=100


communicator = MPI.COMM_WORLD
size_comm = communicator.Get_size()
rank_in_1d = int(np.sqrt(size_comm)) 
if rank_in_1d*rank_in_1d != size_comm:
    print(RuntimeError)
    sys.exit()
rank=communicator.Get_rank()
x=length//(rank_in_1d)+2
y=length//(rank_in_1d)+2
pos_g_x,pos_g_y=rank % int(np.sqrt(size_comm)),rank // int(np.sqrt(size_comm))
l_boundary=pos_g_x==0
b_boundary=pos_g_y==0
r_boundary=pos_g_x==int(np.sqrt(size_comm))-1
t_boundary=pos_g_y==int(np.sqrt(size_comm))-1
t_neighbour=rank+int(np.sqrt(size_comm))
b_neighbour=rank-int(np.sqrt(size_comm))
r_neighbour=rank+1
l_neighbour=rank-1


den=np.ones((x,y))
vel=np.zeros((2,x,y))
f=equilibrium(den,vel)
for step in range(steps):
    f=streaming(f)
    f=boundry_condition(f,lid_vel,b_boundary,t_boundary,r_boundary,l_boundary)
    den=density(f)
    vel=velcoity(f,den)
    f=collision(f,vel,den,relaxation)
    f=communicate(communicator,f,r_boundary,l_boundary,b_boundary,t_boundary,r_neighbour,l_neighbour,b_neighbour,t_neighbour)
f=collapse(f)
den=density(f)
vel=velcoity(f,den)
plotter(steps,length,vel,f,relaxation)

