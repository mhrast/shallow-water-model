"""Created on Mon Oct 31 08:50:47 2022 @author: mhrast."""
# SHALLOW WATER MODEL
# Copyright (c) 2017 by Paul Connolly
#
# Copying and distribution of this file, with or without modification,
# are permitted in any medium without royalty provided the copyright
# notice and this notice are preserved.  This file is offered as-is,
# without any warranty.
#
# This model integrates the shallow water equations in conservative form
# in a channel using the Lax-Wendroff method.  It can be used to
# illustrate a number of meteorological phenomena.

# ------------------------------------------------------------------
import numpy as np
import sys
import scipy.io as sio
from lax_wendroff import lax_wendroff
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.animation as animation
import warnings
import PySimpleGUI as sg
# ------------------------------------------------------------------
# SECTION 0: Definitions (normally don't modify this section)
# ------------------------------------------------------------------

# Possible initial conditions of the height field
UNIFORM_WESTERLY = 1
ZONAL_JET = 2
REANALYSIS = 3
GAUSSIAN_BLOB = 4
STEP = 5
CYCLONE_IN_WESTERLY = 6
SHARP_SHEAR = 7
EQUATORIAL_EASTERLY = 8
SINUSOIDAL = 9

# Possible orographies
FLAT = 0
SLOPE = 1
GAUSSIAN_MOUNTAIN = 2
EARTH_OROGRAPHY = 3
SEA_MOUNT = 4

# ------------------------------------------------------------------
# Graphic User Interface (GUI)
# ------------------------------------------------------------------

sg.theme('Dark Grey 13')

initial_conditions = ['UNIFORM_WESTERLY', 'ZONAL_JET', 'REANALYSIS',
                      'GAUSSIAN_BLOB', 'STEP', 'CYCLONE_IN_WESTERLY',
                      'SHARP_SHEAR', 'EQUATORIAL_EASTERLY', 'SINUSOIDAL']

orographies = ['FLAT', 'SLOPE', 'GAUSSIAN_MOUNTAIN', 'EARTH_OROGRAPHY',
               'SEA_MOUNT']

true_false = ['True', 'False']

layout = [[sg.Text('Configure the model parameters here:')],

          [sg.Text('Initial conditions:'),
           sg.Combo(initial_conditions, key='initial_conditions',
                    default_value=initial_conditions[0],
                    auto_size_text=True, readonly=True)],

          [sg.Text('Orographies:'),
           sg.Combo(orographies, key='orographies',
                    default_value=orographies[0],
                    auto_size_text=True, readonly=True)],

          [sg.Text('Initially geostrophic:'),
           sg.Combo(true_false,
                    default_value=true_false[1], key='initially_geostrophic')],

          [sg.Text('Add random height noise:'),
           sg.Combo(true_false,
                    default_value=true_false[1],
                    key='add_random_height_noise')],

          [sg.Text('Timestep (minutes):'),
           sg.Input(default_text='1', key='dt_mins')],

          [sg.Text('Time between outputs (minutes):'),
           sg.Input(default_text='60', key='output_interval_mins')],

          [sg.Text('Total simulation length (days):'),
           sg.Input(default_text='2', key='forecast_length_days')],

          [sg.Text('g:'), sg.Input(default_text='9.81', key='gravity')],

          [sg.Text('f:'), sg.Input(default_text='1e-4', key='coriolis_mean')],

          [sg.Text('beta:'),
           sg.Input(default_text='1.6e-11', key='coriolis_gradient')],

          [sg.OK(), sg.Cancel()]]

window = sg.Window('Model configuration', layout)

event, values = window.read()
if event == 'Cancel' or event is None:
    window.close()
    sys.exit()

initial_conditions = eval(values['initial_conditions'])
orography = eval(values['orographies'])

initially_geostrophic = bool(values['initially_geostrophic'])
add_random_height_noise = bool(values['add_random_height_noise'])

g = float(values['gravity'])  # Acceleration due to gravity (m/s2)
f = float(values['coriolis_mean'])  # Coriolis parameter (s-1)
beta = float(values['coriolis_gradient'])  # Meridional gradient of f (s-1m-1)

dt_mins = float(values['dt_mins'])  # Timestep (minutes)
# Time between outputs (minutes)
output_interval_mins = float(values['output_interval_mins'])
# Total simulation length (days)
forecast_length_days = float(values['forecast_length_days'])

window.close()

# ------------------------------------------------------------------
# SECTION 1: Configuration
# ------------------------------------------------------------------

print("0. Initializing configuration...")
g = 9.81              # Acceleration due to gravity (m/s2)

f = 1e-4              # Coriolis parameter (s-1)
# f = 0.

beta = 1.6e-11        # Meridional gradient of f (s-1m-1)
# beta = 0.
# beta = 5e-10
# beta = 2.5e-11

dt_mins = 1.                # Timestep (minutes)

output_interval_mins = 60.  # Time between outputs (minutes)

forecast_length_days = 4.   # Total simulation length (days)

initial_conditions = UNIFORM_WESTERLY
orography = EARTH_OROGRAPHY
initially_geostrophic = True     # Can be "True" or "False"
add_random_height_noise = False  # Can be "True" or "False"

# If you change the number of gridpoints then orography=EARTH_OROGRAPHY
# or initial_conditions=REANALYSIS won't work
nx = 254  # Number of zonal gridpoints
ny = 50   # Number of meridional gridpoints

dx = 100.0e3  # Zonal grid spacing (m)
dy = dx       # Meridional grid spacing (m)

# Specify the range of heights to plot in metres
plot_height_range = np.array([9500., 10500.])

# ------------------------------------------------------------------
# SECTION 2: Act on the configuration information
# ------------------------------------------------------------------

dt = dt_mins * 60.0                                  # Timestep (s)
output_interval = output_interval_mins * 60.0        # Time between outputs (s)
forecast_length = forecast_length_days * 24.0 * 3600.0   # Forecast length (s)

nt = int(np.fix(forecast_length / dt) + 1)               # Number of timesteps
timesteps_between_outputs = np.fix(output_interval / dt)
noutput = int(np.ceil(nt /
                      timesteps_between_outputs))    # Number of output frames

x = np.mgrid[0:nx] * dx       # Zonal distance coordinate (m)
y = np.mgrid[0:ny] * dy       # Meridional distance coordinate (m)
Y, X = np.meshgrid(y, x)  # Create matrices of the coordinate variables


# Create the orography field "H"
if orography == FLAT:
    H = np.zeros((nx, ny))

elif orography == SLOPE:
    H = 9000. * 2. * np.abs((np.mean(x) - X) / np.max(x))

elif orography == GAUSSIAN_MOUNTAIN:
    std_mountain_x = 5. * dx  # Std. dev. of mountain in x direction (m)
    std_mountain_y = 5. * dy  # Std. dev. of mountain in y direction (m)
    H = 4000.*np.exp(-0.5 * ((X - np.mean(x)) / std_mountain_x)**2
                     - 0.5 * ((Y - np.mean(y)) / std_mountain_y)**2)

elif orography == SEA_MOUNT:
    std_mountain = 40.0*dy  # Standard deviation of mountain (m)
    H = 9250.*np.exp(-((X - np.mean(x))**2
                     + (Y - 0.5 * np.mean(y))**2) / (2 * std_mountain**2))

elif orography == EARTH_OROGRAPHY:
    mat_contents = sio.loadmat("digital_elevation_map.mat")
    H = mat_contents["elevation"]
    # Enforce periodic boundary conditions in x
    H[[0, -1], :] = H[[-2, 1], :]

else:
    print("Don't know what to do with orography=" + np.num2str(orography))
    sys.exit()


# Create the initial height field
if initial_conditions == UNIFORM_WESTERLY:
    mean_wind_speed = 20.  # m/s
    height = 10000. - (mean_wind_speed*f/g) * (Y - np.mean(y))

elif initial_conditions == SINUSOIDAL:
    height = 10000. - 350. * np.cos(Y/np.max(y) * 4*np.pi)

elif initial_conditions == EQUATORIAL_EASTERLY:
    height = 10000. - 50. * np.cos((Y - np.mean(y)) * 4*np.pi/np.max(y))

elif initial_conditions == ZONAL_JET:
    height = 10000. - 400. * np.tanh(20.0 * ((Y - np.mean(y))/np.max(y)))

elif initial_conditions == REANALYSIS:
    mat_contents = sio.loadmat("reanalysis.mat")
    pressure = mat_contents["pressure"]
    height = 0.99 * pressure / g

elif initial_conditions == GAUSSIAN_BLOB:
    std_blob = 8.0 * dy  # Standard deviation of blob (m)
    height = 9750. + 1000. * np.exp(-((X - 0.25*np.mean(x))**2
                                    + (Y - np.mean(y))**2)/(2 * std_blob**2))

elif initial_conditions == STEP:
    height = 9750. * np.ones((nx, ny))
    height[np.where((X < np.max(x)/5.) & (Y > np.max(y)/10.) &
                    (Y < np.max(y)*0.9))] = 10500.

elif initial_conditions == CYCLONE_IN_WESTERLY:
    mean_wind_speed = 20.  # m/s
    std_blob = 7.0 * dy  # Standard deviation of blob (m)
    height = 10000. - (mean_wind_speed*f/g) * (Y - np.mean(y)) \
        - 500. * np.exp(-((X - 0.5*np.mean(x))**2
                        + (Y - np.mean(y))**2)/(2 * std_blob**2))
    max_wind_speed = 20.  # m/s
    height = 10250. - (max_wind_speed*f/g) * (Y - np.mean(y))**2/np.max(y) \
        - 1000.*np.exp(-(0.25*(X - 1.5*np.mean(x))**2
                       + (Y - 0.5*np.mean(y))**2)/(2 * std_blob**2))

elif initial_conditions == SHARP_SHEAR:
    mean_wind_speed = 50.  # m/s
    height = (mean_wind_speed * f / g) * np.abs(Y - np.mean(y))
    height = 10000. + height - np.mean(height[:])

else:
    print("Don't know what to do with initial_conditions=%f"
          % initial_conditions)
    sys.exit()


# Coriolis parameter as a matrix of values varying in y only
F = f + beta * (Y - np.mean(y))

# Initialize the wind to rest
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))

# We may need to add small-amplitude random noise in order to initialize
# instability
if add_random_height_noise:
    r, c = np.shape(height)
    height += 1.0 * np.random.randn(r, c) * (dx/1.0e5) * (np.abs(F) / 1e-4)


if initially_geostrophic:
    # Centred spatial differences to compute geostrophic wind
    # ug = -g/f * dz/dy, vg = g/f * dz/dx
    u[:, 1:-1] = -(0.5 * g / (F[:, 1:-1])) * \
        (height[:, 2:] - height[:, 0:-2]) / dy

    v[1:-1, :] = (0.5 * g / (F[1:-1, :])) * \
        (height[2:, :] - height[0:-2, :]) / dx

    # Zonal wind is periodic so set u(1) and u(end) as dummy points that
    # replicate u(end-1) and u(2), respectively
    u[[0, -1], :] = u[[1, -2], :]
    # Meridional wind must be zero at the north and south edges of the channel
    v[:, [0, -1]] = 0.

    # Don't allow the initial wind speed to exceed 200 m/s anywhere
    max_wind = 200.
    u[np.where(u > max_wind)] = max_wind
    u[np.where(u < -max_wind)] = -max_wind
    v[np.where(v > max_wind)] = max_wind
    v[np.where(v < -max_wind)] = -max_wind


# Define h as the depth of the fluid (whereas "height" is the height of
# the upper surface)
h = height - H

# Initialize the 3D arrays where the output data will be stored
u_save = np.zeros((nx, ny, noutput))
v_save = np.zeros((nx, ny, noutput))
h_save = np.zeros((nx, ny, noutput))
t_save = np.zeros((noutput, 1))

vorticity = np.zeros(np.shape(u))
vorticity_save = np.zeros(np.shape(u_save))

# Index to stored data
i_save = 0

# ------------------------------------------------------------------
# SECTION 3: Main loop
# ------------------------------------------------------------------
print()
print("1. Initializing main loop...")
print()
for n in range(0, nt):
    # Every fixed number of timesteps we store the fields
    if np.mod(n, timesteps_between_outputs) == 0:

        max_u = np.sqrt(np.max(u[:] * u[:] + v[:] * v[:]))

        print("Time = %.2f hours (max %.2f); max(|u|) = %.2f"
              % (n * dt / 3600., forecast_length_days * 24., max_u))

        u_save[:, :, i_save] = u
        v_save[:, :, i_save] = v
        h_save[:, :, i_save] = h
        vorticity_save[:, :, i_save] = vorticity
        t_save[i_save] = n * dt
        i_save += 1

    # Compute the accelerations
    u_accel = F[1:-1, 1:-1] * v[1:-1, 1:-1] \
        - (g/(2.*dx)) * (H[2:, 1:-1] - H[0:-2, 1:-1])
    v_accel = -F[1:-1, 1:-1] * u[1:-1, 1:-1] \
        - (g/(2.*dy)) * (H[1:-1, 2:] - H[1:-1, 0:-2])

    # Compute the vorticity
    vorticity[1:-1, 1:-1] = (1. / dy)*(u[1:-1, 0:-2] - u[1:-1, 2:]) \
        + (1. / dx) * (v[2:, 1:-1] - v[0:-2, 1:-1])

    # Call the Lax-Wendroff scheme to move forward one timestep
    unew, vnew, h_new = lax_wendroff(dx, dy, dt,
                                     g, u, v, h, u_accel, v_accel)

    # Update the wind and height fields, taking care to enforce
    # boundary conditions
    u[1:-1, 1:-1] = unew
    v[1:-1, 1:-1] = vnew

    # periodic boundary conditions in x direction for u
    # first x-slice
    u[0, 1:-1] = unew[-1, :]
    u[0, 0] = unew[-1, 0]
    u[0, -1] = unew[-1, -1]
    v[0, 1:-1] = vnew[-1, :]
    v[0, 0] = vnew[-1, 0]
    v[0, -1] = vnew[-1, -1]
    # last x-slice
    u[-1, 1:-1] = unew[1, :]
    u[-1, 0] = unew[1, 0]
    u[-1, -1] = unew[1, -1]
    v[-1, 1:-1] = vnew[1, :]
    v[-1, 0] = vnew[1, 0]
    v[-1, -1] = vnew[1, -1]

    # periodic boundary conditions in x direction for h
    # interior
    h[1:-1, 1:-1] = h_new
    # first x-slice
    h[0, 1:-1] = h_new[-1, :]
    # last x-slice
    h[-1, 1:-1] = h_new[1, :]

    # boundary conditions in y direction for v
    # no flux from north / south
    v[:, [0, -1]] = 0.


# ------------------------------------------------------------------
# SECTION 4: Animation
# ------------------------------------------------------------------
print()
print('2. Initializing animation...')
print()

# This animates the height field and the vorticity produced by
# a shallow water model. Press space to pause and use the arrow keys
# to change between single frames.

# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'],
#    'size': 22})
# # for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)


fig = plt.figure(figsize=(16, 12), dpi=100)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

title_size = 18
label_size = 14
cbar_size = 14

ax1.autoscale(enable=True, axis='y', tight=True)

# Axis units are thousands of kilometers (x and y are in metres)
x_1000km = x * 1e-6
y_1000km = y * 1e-6

# Set colormap to have 64 entries
ncol = 64

# Interval between arrows in the velocity vector plot
interval = 6

# Decide whether to show height in metres or km
if np.mean(plot_height_range) > 1000:
    height_scale = 0.001
    height_title = 'Height (km)'
else:
    height_scale = 1
    height_title = 'Height (m)'


print('Maximum orography height = %.2f m' % np.max(H))

# Extract the height and velocity components for this frame
h = np.squeeze(h_save[:, :, 0])
u = np.squeeze(u_save[:, :, 0])
v = np.squeeze(v_save[:, :, 0])
vorticity = np.squeeze(vorticity_save[:, :, 0])

# Compute the vorticity omega = du/dy - dv/dx
# vorticity[1:-1, 1:-1] = (u[1:-1, 0:-2] - u[1:-1, 2:]) / dy \
#                      + (v[2:, 1:-1] - v[0:-2, 1:-1]) / dx

# cmap="jet"

# Plot the height field
im1 = ax1.imshow(np.transpose(h + H) * height_scale,
                 extent=[np.min(x_1000km), np.max(x_1000km),
                 np.min(y_1000km), np.max(y_1000km)],
                 cmap="seismic", origin="lower")

# Set other axes properties and plot a colorbar
cb1 = fig.colorbar(im1, ax=ax1)
cb1.set_label("height (km)", fontsize=cbar_size)
# Contour the terrain:

# This code line removes the warning message when the orography is flat
warnings.filterwarnings("ignore", message="No contour levels were found "
                        "within the data range.")

cs = ax1.contour(x_1000km, y_1000km, np.transpose(H),
                 levels=range(1, 11001, 1000), colors='k')

cs2 = ax2.contour(x_1000km, y_1000km, np.transpose(H),
                  levels=range(1, 11001, 1000), colors='k')

quiver_scale = 1

# Plot the velocity vectors
Q = ax1.quiver(x_1000km[2::interval], y_1000km[2::interval],
               np.transpose(u[2::interval, 2::interval]),
               np.transpose(v[2::interval, 2::interval]),
               angles='xy', scale_units='xy', scale=1e2, pivot='tip',
               headlength=5*quiver_scale, headwidth=3*quiver_scale,
               headaxislength=4.5*quiver_scale, width=1.3e-3)

# pivot='tail', scale=5e2
ax1.set_xlabel("X distance ($10^3$ km)", fontsize=label_size)
ax1.set_ylabel("Y distance ($10^3$ km)", fontsize=label_size)
ax1.set_title(height_title, fontsize=title_size)
tx1 = ax1.text(0, np.max(y_1000km)*1.02, "Time = %.1f hours"
               % (np.squeeze(t_save[0]) / 3600.))

# Now plot the vorticity
# np.transpose(vorticity)
im2 = ax2.imshow(np.transpose(vorticity),
                 extent=[np.min(x_1000km), np.max(x_1000km),
                 np.min(y_1000km), np.max(y_1000km)],
                 cmap="seismic", origin="lower")

# Set other axes properties and plot a colorbar
cb2 = fig.colorbar(im2, ax=ax2)
cb2.set_label("vorticity (s$^{-1}$)", fontsize=cbar_size)

ax2.set_xlabel("X distance ($10^3$ km)", fontsize=label_size)
ax2.set_ylabel("Y distance ($10^3$ km)", fontsize=label_size)
ax2.set_title("Relative vorticity (s$^{-1}$)", fontsize=title_size)
tx2 = ax2.text(0, np.max(y_1000km)*1.02, "Time = %.1f hours"
               % (np.squeeze(t_save[0])/3600.))

im1.set_clim((plot_height_range * height_scale))
im2.set_clim((-3e-4, 3e-4))
ax1.axis((0., np.max(x_1000km), 0., np.max(y_1000km)))
ax2.axis((0., np.max(x_1000km), 0., np.max(y_1000km)))


ani_running = True

# This section sets up the plot to use arrow keys to select frames and
# the space bar to pause the animation


def update_time():
    """Yield the frame number based on the maximum output of time steps.

    Returns
    -------
    t : iterator object
    """
    t = 0
    t_max = noutput
    while t < t_max - 1:
        t += ani.direction
        yield t


def on_press(event):
    """Allow the animation to be paused and fast-forwarded or rewinded.

    Use the space key to pause the animation and the left/right arrow keys
    to fast-forward/rewind the animation.

    Returns
    -------
    None
    """
    if event.key.isspace():
        if ani.running:
            ani.pause()
        else:
            ani.resume()
        ani.running ^= True

    elif event.key == 'left':
        ani.direction = -1

    elif event.key == 'right':
        ani.direction = 1

    # Manually update the plot
    if event.key in ['left', 'right']:
        t = ani.frame_seq.__next__()
        update_plot(t)
        plt.draw()


def update_plot(it):
    """Update the figure for matplotlib's FuncAnimation.

    Parameters
    ----------
    it : integer
        number of frame

    Returns
    -------
    im1, im2, cs, cs2, tx1, tx2, Q, : matplotlib objects
        The objects required to update the plot
    """
    # Extract the height and velocity components for this frame
    h = np.squeeze(h_save[:, :, it])
    u = np.squeeze(u_save[:, :, it])
    v = np.squeeze(v_save[:, :, it])
    vorticity = np.squeeze(vorticity_save[:, :, it])

    # Compute the vorticity; already computed
    # vorticity[1:-1, 1:-1] = (1./dy)*(u[1:-1, 0:-2]-u[1:-1, 2:]) \
    #    + (1./dx)*(v[2:, 1:-1]-v[0:-2, 1:-1])

    # top plot:
    im1.set_data(np.transpose(H + h) * height_scale)

    # CAUTION! set_array() does not work anymore! Contour has to be drawn
    # new each time the plot updates

    # cs.set_array(np.transpose(h))
    # cs2.set_array(np.transpose(h))

    global cs, cs2

    # for i in cs.collections:
    #    i.remove()

    #cs.remove()
    #cs = ax1.contour(x_1000km, y_1000km, np.transpose(h),
                     #levels=range(1, 11001, 1000), colors='k')

    # for i in cs2.collections:
    #    i.remove()

    #cs2.remove()
    #cs2 = ax2.contour(x_1000km, y_1000km, np.transpose(h),
                      #levels=range(1, 11001, 1000), colors='k')

    Q.set_UVC(np.transpose(u[2::interval, 2::interval]),
              np.transpose(v[2::interval, 2::interval]))
    tx1.set_text('Time = %.1f hours' % (np.squeeze(t_save[it])/3600.))

    # bottom plot:
    im2.set_data(np.transpose(vorticity))
    tx2.set_text('Time = %.1f hours' % (np.squeeze(t_save[it])/3600.))

    im1.set_clim((plot_height_range * height_scale))
    im2.set_clim((-3e-4, 3e-4))
    ax1.axis((0., np.max(x_1000km), 0., np.max(y_1000km)))
    ax2.axis((0., np.max(x_1000km), 0., np.max(y_1000km)))

    return im1, im2, tx1, tx2, Q,


fig.canvas.mpl_connect('key_press_event', on_press)

# Setting frames to the size of the vector removes the IndexError in console
# 60 Hz monitor => 0.0167 s = 167 ms interval

ani = animation.FuncAnimation(fig, update_plot, frames=update_time,
                              interval=167, repeat_delay=1000,
                              cache_frame_data=False)

ani.running = True
ani.direction = 1
print()
print("Successfully finished")
plt.show()

# possibility to provide (254, 50) data of elevation or pressure map
elev = sio.loadmat('digital_elevation_map.mat')['elevation']
rean = sio.loadmat('reanalysis')['pressure']

# filename = r"vid/swe_animation.mp4"
# writervideo = animation.FFMpegWriter(fps=30)
# ani.save(filename, writer=writervideo)
