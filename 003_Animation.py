import numpy as np
import matplotlib.pyplot as plt
# plate size, mm
w = h = 10.
# intervals in x-, y- directions, mm
dx = dy =0.1
# Thermal diffusivity of steel, mm2.s-1
D = 100.

Tcool, Thot = 0, 100

nx, ny = int(w/dx), int(h/dy)

dx2, dy2 = dx*dx, dy*dy
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))
u0 =np.ones((nx, ny))
u = np.empty((nx, ny))

# Initial conditions - ring of inner radius r, width dr centred at (cx,cy) (mm)
r, cx, cy = 4, 5, 5
r2 = r**2
for i in range(nx):
    for j in range(ny):
        p2 = (i*dx-cx)**2 + (j*dy-cy)**2
        if p2 < r2:
           u0[i,j] = Thot
def do_timestep(u0, u):
 # Propagate with forward-difference in time, central-difference in space
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * ((u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2 + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2 )
    u0 = u.copy()
    return u0, u
# Number of timesteps
x = np.linspace(0,dx,nx)
y = np.linspace(0,dy,ny)
XX, YY = np.meshgrid(x, y)
v = np.linspace(Tcool, Thot, 100, endpoint=True)
plt.ion()
plt.rcParams['image.cmap'] = 'jet'
fig, ax = plt.subplots()
for t in range(1000):
    u0, u = do_timestep(u0, u)
    # print (u)
    plt.cla()
    CF = ax.contourf(XX, YY, u,levels = v)
    # ax.set_title('Time (sec): %.2f' %te)
    # ax.set_xlim([0,11])
    # ax.set_ylim([0,11])
    ax.set_xlabel('Distance(km)')
    ax.set_ylabel('Distance(km)')
    ax.set_aspect('equal')
    fig.canvas.draw()