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
# Calculate equilibrium distribution function
def equilibrium(d,v):
    vel_mag = v[0,:,:] ** 2 + v[1,:,:] ** 2
    temp = np.dot(v.T,c.T).T
    sq_velocity = temp ** 2
    f_eq = (((1 + 3*(temp) + 9/2*(sq_velocity) - 3/2*(vel_mag)) * d ).T * w).T
    return f_eq

# Perform streaming step, shifting distribution functions across the lattice
def streaming(f):
    for i in range(9):
        f[i,:,:] = np.roll(f[i,:,:],c[i], axis = (0,1))
    return f

# Apply boundary conditions for coutte 
def boundry_condition(f,lid_vel):
    f[2, :, 1] = f[4, :, 0] 
    f[5, :, 1] = f[7, :, 0]
    f[6, :, 1] = f[8, :, 0]
    f[4, :, -2] = f[2, :, -1]
    f[7, :, -2] = f[5, :, -1] - 1 / 6 * lid_vel
    f[8, :, -2] = f[6, :, -1] + 1 / 6 * lid_vel
    return f

# Simulation parameters
x=100
y=50
steps=3000
rate=100
lid_vel=0.1

# Initial conditions for density and velocity
den=np.ones((x,y+2))
vel=np.zeros((2,x,y+2))    
f=equilibrium(den,vel)

# Main simulation loop
for step in range(steps):
    den=density(f)
    vel=velcoity(f,den)
    f=collision(f,vel,den)
    f=streaming(f)
    f=boundry_condition(f,lid_vel)
    if (step % rate) == 0 and step > 0:
                rate = rate*2
                plt.plot(vel[0, 50, 1:-1],np.arange(0,50))
plt.plot(vel[0, 50, 1:-1],np.arange(0,50),label = "Simulated Velocity")
Analytical_vel = lid_vel*1/50*np.arange(0,50)
plt.plot(Analytical_vel,np.arange(0,50), label ="Analytical Velocity")
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.title('Couette flow')
min_x,max_x,min_y,max_y = plt.axis()
plt.plot([min_x,max_x],[min_y+2,min_y+2], color='k',label="Rigid Wall")
plt.plot([min_x,max_x],[max_y-2,max_y-2], color='r',label="Moving Wall")
plt.legend()
plt.savefig("Task_4_Coutte.png")
plt.show()
