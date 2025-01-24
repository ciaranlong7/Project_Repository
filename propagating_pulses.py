import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import PolyCollection
from scipy.fft import fft, ifft, fftfreq

#E_z_t0: An array of the input electric field
#t: The time at which the electric field is calculated
#dz: Spatial resolution
def electric_field_tz(E_z_t0, t, dz):

    # Compute the Fourier transform of the input electric field
    E_k = fft(E_z_t0)

    # Number of spatial points
    N = len(E_z_t0)

    # Wave numbers corresponding to Fourier transform
    w_k = fftfreq(N, dz)*2*np.pi 

    E_k_t = E_k * np.exp(-1j*w_k*t)

    # Inverse Fourier transform to get E(z, t) - final step of hedgehog in time equation.
    E_z_t = ifft(E_k_t)

    modulus_E_z_t = np.sqrt(np.real(E_z_t)**2+ np.imag(E_z_t)**2)

    return modulus_E_z_t

####Different possible inputs below:
def gauss(z_points, b, a):
    #a cannot equal 0
    E_z_t0 = np.array([np.exp(-((z-b)**2)/(a**2))for z in z_points])
    return E_z_t0

def rect(z_points, min, max, height):
    E_z_t0 = np.where(np.logical_and(z_points >= min, z_points <= max), height, 0)
    return E_z_t0

dz = 0.01  #Spatial resolution (how far apart points are plotted)
z_points = np.arange(-3, 20, dz)  # min,max of z-axis for the plot

E_z_t0 = gauss(z_points, 0, 1) 
E_z_t0 = rect(z_points, 0, 1, 5)

t = 0.1
E_z_t = electric_field_tz(E_z_t0, t, dz) #Electric field after time t


#Making a plot:
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25) #making space for slider

ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor="lightgray")
slider = Slider(ax_slider, "Time (t)", valmin=0, valmax=10, valinit=t, valstep=0.01)

ax.plot(z_points, E_z_t0, label="E(z, t=0)")
ax.fill_between(z_points, E_z_t0, color="#1f77b4", alpha=0.3)
line, = ax.plot(z_points, E_z_t, color="#ff7f0e", label=f"E(z, t={t})")
fill = ax.fill_between(z_points, E_z_t, color="#ff7f0e", alpha=0.3)
ax.set_xlabel("Z")
ax.set_ylabel("Electric Field")
ax.set_title("Electric Field Evolution")
ax.legend()

def update(val):
    t = slider.val
    E_z_t = electric_field_tz(E_z_t0, t, dz)
    line.set_ydata(E_z_t)
    for collection in [c for c in ax.collections if isinstance(c, PolyCollection)]:
        collection.remove()
    ax.fill_between(z_points, E_z_t0, color="#1f77b4", alpha=0.3)
    ax.fill_between(z_points, E_z_t, color="#ff7f0e", alpha=0.3)
    line.set_label(f"E(z, t={t:.2f})")
    ax.legend()
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()