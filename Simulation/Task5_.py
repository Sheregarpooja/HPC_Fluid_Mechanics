import numpy as np
import matplotlib.pyplot as plt

# Define lattice velocity vectors for D2Q9 model
c = np.array([[ 0, 1, 0,-1, 0, 1,-1,-1, 1],[ 0, 0, 1, 0,-1, 1, 1,-1,-1]]).T
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]).T


# Compute the density at each lattice point by summing over velocity directions 
def density(f):
    a=np.sum(f, axis=0) 
    return a
# Compute the velocity at each lattice point
def velcoity(f,d):
    return np.dot(f.T,c).T/d

# Perform collision step, adjusting distribution functions towards equilibrium
def collision(f,v,d,relaxation=0.5):
    eq = equilibrium(d,v)
    f -= relaxation * (f-eq)    
    return f
#Calculate equilibrium distribution for Poiseuille flow 
def equilibrium_pois(d,v):
    vel_mag =  v[0,:] ** 2 + v[1,:] ** 2
    temp = np.dot(v.T,c.T).T
    sq_velocity = temp ** 2
    f_eq =( ((1 + 3*(temp) + 9/2*(sq_velocity) - 3/2*(vel_mag)) * d).T * w).T
    return f_eq
#general
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
#  bounce-back boundary conditions 
def boundry_condition(f):
    f[2, :, 1] = f[4, :, 0] 
    f[5, :, 1] = f[7, :, 0]
    f[6, :, 1] = f[8, :, 0]
    f[4, :, -2] = f[2, :, -1]
    f[7, :, -2] = f[5, :, -1] 
    f[8, :, -2] = f[6, :, -1] 
    return f

#  periodic boundary conditions with pressure difference
def periodic_conditions(den,vel,den_in,den_out,f):
        equi = equilibrium(den,vel)
        equi_in = equilibrium_pois(den_in, vel[:,-2,:])
        f[:, 0, :] = equi_in + (f[:, -2, :] - equi[:, -2, :])
        equi_out = equilibrium_pois(den_out, vel[:, 1, :])
        f[:, -1, :] = equi_out + (f[:, 1, :] - equi[:, 1, :])
        return f

x=100
y=50
steps=5000
rate=100
diff=0.001 #pressue difference at wall

# Initialize density, velocity, and distribution function

den_in = 1+diff 
den_out = 1-diff
relaxation=0.5
shear_viscosity = (1/relaxation-0.5)/3
den=np.ones((x,y+2))
vel=np.zeros((2,x,y+2))    
f=equilibrium(den,vel)
for step in range(steps):
    den=density(f)
    vel=velcoity(f,den)
    f=periodic_conditions(den,vel,den_in,den_out,f)
    f=streaming(f)
    f=boundry_condition(f)
    f=collision(f,vel,den)
    
    if (step % rate) == 0 and step > 0:
                rate = rate*2
                plt.plot(vel[0, 50, 1:-1],np.arange(0,50))
plt.plot(vel[0, 50, 1:-1],np.arange(0,50),label = "Simulated Velocity")
delta = 2.0 * diff /x / shear_viscosity / 2.
yi = np.linspace(0, y, y+1) + 0.5
Analytical = delta * yi * (y - yi) / 3.
plt.plot(Analytical[:-1],np.arange(0,50), label='Analytical Velocity')
min_x,max_x,min_y,max_y = plt.axis()
plt.plot([min_x,max_x],[min_y+2,min_y+2], color='k', label="Rigid Wall")
plt.plot([min_x,max_x],[max_y-2,max_y-2], color='k')

plt.xlabel('Position')
plt.ylabel('Velocity')
plt.title('Poiseuille flow')
plt.legend(loc='lower right')
plt.savefig("Task_5_Poiseuille.png")
plt.show()
