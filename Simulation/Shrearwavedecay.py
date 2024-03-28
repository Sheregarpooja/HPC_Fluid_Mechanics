import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.interpolate import make_interp_spline

c = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1], [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]).T

ep = 0.05
omega = 0.5


def density(f):
    a = np.sum(f, axis=0)
    return a


def velcoity(f, d):
    return np.dot(f.T, c).T / d


def v_abs(v):
    vx = v[0, :, 1:-1]
    print(np.shape(vx))
    vy = v[1, :, 1:-1]
    print(np.shape(vy))
    sq_velocity = np.sqrt(vx ** 2 + vy ** 2)
    return sq_velocity


def collision(f, v, d, relaxation=0.5):
    eq = equilibrium(d, v)
    f -= relaxation * (f - eq)
    return f



def equilibrium(d, v):
    vel_mag = v[0, :, :] ** 2 + v[1, :, :] ** 2
    temp = np.dot(v.T, c.T).T
    sq_velocity = temp ** 2
    f_eq = (((1 + 3 * (temp) + 9 / 2 * (sq_velocity) - 3 / 2 * (vel_mag)) * d).T * w).T
    return f_eq

def calculate_collision(f, rl):
    dens = density(f)
    v = velcoity(f, dens)
    f_eq = equilibrium(den, v)
    f -= rl * (f - f_eq)
    return f, dens, v

def streaming(f):
    for i in range(9):
        f[i, :, :] = np.roll(f[i, :, :], c[i], axis=(0, 1))
    return f


def theoretical_velocity(v, y):
    decay_factor = (2 * np.pi / y) ** 2
    y = np.exp(-v * decay_factor)
    return y


def decay_perturbation(time, viscosity, y):
    decay_rate = (2 * np.pi / y) ** 2
    decay = ep * np.exp(-viscosity * decay_rate * time)
    return decay


x = 100
y = 100
steps = 1000
rate = 20
den = np.ones((x, y + 2))
vel = np.zeros((2, x, y + 2))
vel[1, :, :] = ep * np.sin(2 * np.pi / y * np.arange(x)[:, np.newaxis])
f = equilibrium(den, vel)
ep = 0.05
max_vel = []
min_vel = []
vel_analytical = []
ve_decay = []
tempv = vel
for cur_step in range(steps):
    f = streaming(f)
    #den = density(f)
    #vel = velcoity(f, den)
    #f = collision(f, vel, den)
    f,den,vel = calculate_collision(f,omega)

    if cur_step % rate == 0:
        plt.clf()
        mesh_x, mesh_y = np.meshgrid(np.arange(y), np.arange(x))
        velocity_magnitude = v_abs(vel)
        color_map = plt.scatter(mesh_x, mesh_y, c=velocity_magnitude.T, vmin=np.min(velocity_magnitude.T),
                                vmax=np.max(velocity_magnitude.T))
        plt.colorbar(color_map, label="Velocity", pad=0.15).set_label("Velocity", rotation=270, labelpad=15)
        plt.title(f"Shear wave step {cur_step}")
        plt.xlabel("x-Position")
        plt.ylabel("y-Position")
        plt.savefig(f'graphs3.1/swave/shear wave step {cur_step}.png')

    kinematic_viscosity = (1 / omega - 0.5) / 3
    y_val = theoretical_velocity(kinematic_viscosity, y)
    y_val = y_val * tempv[1, x // 4, :]
    tempv[1, x // 4, :] = y_val
    vel_analytical.append(max(y_val))

    max_vel.append(vel[1, :, :].max())
    min_vel.append(vel[1, :, :].min())

    if cur_step % rate == 0:
        y_positions = np.arange(y)
        specific_velocity = vel[1, x // 4, 1:-1]
        print(np.shape(specific_velocity))
        print(np.shape(y_positions))
        plt.clf()
        plt.ylim([-ep, ep])  # Set the limits of y-axis to [-ep, ep]
        plt.plot(y_positions, specific_velocity, label=f'Step {cur_step}')
        plt.xlabel('Y Position')
        plt.ylabel(f'Velocity u(x = {x // 4}, y)')
        plt.grid(True)
        plt.legend()
        plt.title(f'Velocity Profile at x = {x // 4}, Step {cur_step}')
        plt.savefig(f'graphs3.1/vprofile/Velocity_Profile_Step_{cur_step}.png')
        ve_decay.append(specific_velocity)

plt.figure()
plt.ylim([-ep, ep])

y_positions = np.arange(y)
for index, data in enumerate(ve_decay):
    opacity = (index + 1) / len(ve_decay)
    plt.plot(y_positions, data, alpha=opacity, linewidth=1.5, label=f'Step {index}')
plt.xlabel('Y Position')
plt.ylabel(f'Velocity Profile for u(x = {x // 4})')
plt.title('Combined Velocity Decay Over Time')
plt.grid(True)
plt.savefig('graphs3.1/velocity_decay_over_time.png', bbox_inches='tight')
plt.close()
