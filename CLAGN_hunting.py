import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import pandas as pd
import math
from scipy.interpolate import interp1d
from scipy.stats import median_abs_deviation
from astropy.io import fits
from astropy import units as u #In Astropy, a Quantity object combines a numerical value (like a 1D array of flux) with a physical unit (like W/m^2, erg/s, etc.)
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.coordinates import SkyCoord
import sfdmap
from astroquery.ipac.irsa import Irsa
from dust_extinction.parameter_averages import G23
from astropy.io.fits.hdu.hdulist import HDUList
from astroquery.sdss import SDSS
from sparcl.client import SparclClient
from requests.exceptions import ConnectTimeout
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

c = 299792458

#G23 dust extinction model:
#https://dust-extinction.readthedocs.io/en/latest/api/dust_extinction.parameter_averages.G23.html#dust_extinction.parameter_averages.G23

object_name = '111938.02+513315.5'

def flux(mag, k, wavel): # k is the zero magnitude flux density. For W1 & W2, taken from a data table on the search website - https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
    k = (k*(10**(-6))*(c*10**(10)))/(wavel**2) # converting from Jansky to 10-17 ergs/s/cm2/Å. Express c in Angstrom units
    return k*10**(-mag/2.5)

W1_k = 309.540 #Janskys. This means that mag 0 = 309.540 Janskys at the W1 wl.
W2_k = 171.787
W1_wl = 3.4e4 #Angstroms
W2_wl = 4.6e4

parent_sample = pd.read_csv('clean_parent_sample_no_CLAGN.csv')
Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")
object_data = parent_sample[parent_sample.iloc[:, 3] == object_name]

if len(object_data) == 0: #If a CLAGN; CLAGN are not in parent sample
    parent_sample = pd.read_csv('guo23_parent_sample_no_duplicates.csv')
    object_data = parent_sample[parent_sample.iloc[:, 3] == object_name]
    SDSS_RA = object_data.iloc[0, 0]
    SDSS_DEC = object_data.iloc[0, 1]
    SDSS_plate_number = object_data.iloc[0, 4]
    SDSS_plate = f'{SDSS_plate_number:04}'
    SDSS_fiberid_number = object_data.iloc[0, 6]
    SDSS_fiberid = f"{SDSS_fiberid_number:04}"
    SDSS_mjd = object_data.iloc[0, 5]
    DESI_mjd = object_data.iloc[0, 11]
    SDSS_z = object_data.iloc[0, 2]
    DESI_z = object_data.iloc[0, 9]
    DESI_name = object_data.iloc[0, 10]
else:
    SDSS_RA = object_data.iloc[0, 0]
    SDSS_DEC = object_data.iloc[0, 1]
    SDSS_plate_number = object_data.iloc[0, 4]
    SDSS_plate = f'{SDSS_plate_number:04}'
    SDSS_fiberid_number = object_data.iloc[0, 6]
    SDSS_fiberid = f"{SDSS_fiberid_number:04}"
    SDSS_mjd = object_data.iloc[0, 5]
    DESI_mjd = object_data.iloc[0, 11]
    SDSS_z = object_data.iloc[0, 2]
    DESI_z = object_data.iloc[0, 9]

coord = SkyCoord(SDSS_RA, SDSS_DEC, unit='deg', frame='icrs') #This works

spec_file_path = f'C:/Users/ciara/Dropbox/University/University Work/Fourth Year/Project/Optical Data/{object_name}.fits'
#doesn't work
with fits.open(spec_file_path) as hdul:
    subset = hdul[1]

    sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
    sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms

sfd = sfdmap.SFDMap('SFD_dust_files') #called SFD map, but see - https://github.com/kbarbary/sfdmap/blob/master/README.md
# It explains how "By default, a scaling of 0.86 is applied to the map values to reflect the recalibration by Schlafly & Finkbeiner (2011)"
ebv = sfd.ebv(coord)
print(f"E(B-V): {ebv}")

ext_model = G23(Rv=3.1) #Rv=3.1 is typical for MW - Schultz, Wiemer, 1975
# uncorrected_SDSS = sdss_flux
inverse_SDSS_lamb = [1/(x*10**(-4)) for x in sdss_lamb] #need units of inverse microns for extinguishing
# inverse_DESI_lamb = [1/(x*10**(-4)) for x in desi_lamb]
sdss_flux = sdss_flux/ext_model.extinguish(inverse_SDSS_lamb, Ebv=ebv) #divide to remove the effect of dust
# desi_flux = desi_flux/ext_model.extinguish(inverse_DESI_lamb, Ebv=ebv)

# Correcting for redshift
if object_name in Guo_table4.iloc[:, 0].values:
    object_row = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
    redshift = object_row.iloc[0, 3]
    SDSS_z = redshift
    DESI_z = redshift

sdss_lamb = (sdss_lamb/(1+SDSS_z))
# desi_lamb = (desi_lamb/(1+DESI_z))

print(f'Object Name = {object_name}')
print(f'SDSS Redshift = {SDSS_z}')
print(f'DESI Redshift = {DESI_z}')

#Plotting:
gaussian_kernel = Gaussian1DKernel(stddev=3)

# Smooth the flux data using the Gaussian kernel
if len(sdss_flux) > 0:
    Gaus_smoothed_SDSS = convolve(sdss_flux, gaussian_kernel)
else:
    Gaus_smoothed_SDSS = []
# if len(desi_flux) > 0:
#     Gaus_smoothed_DESI = convolve(desi_flux, gaussian_kernel)
# else:
#     Gaus_smoothed_DESI = []
#BELs
H_alpha = 6562.819
H_beta = 4861.333
Mg2 = 2795.528
C3_ = 1908.734
C4 = 1548.187
Ly_alpha = 1215.670
Ly_beta = 1025.722
#NEL
_O3_ = 5006.843 #underscores indicate square brackets
if len(sdss_lamb) > 0:
    SDSS_min = min(sdss_lamb)
    SDSS_max = max(sdss_lamb)
else:
    SDSS_min = 0
    SDSS_max = 1
# if len(desi_lamb) > 0:
#     DESI_min = min(desi_lamb)
#     DESI_max = max(desi_lamb)
# else:
#     DESI_min = 0
#     DESI_max = 1

# common_ymin = 0
# if len(sdss_flux) > 0 and len(desi_flux) > 0:
#     common_ymax = 1.05*max(Gaus_smoothed_SDSS.tolist()+Gaus_smoothed_DESI.tolist())
# elif len(sdss_flux) > 0:
#     common_ymax = 1.05*max(Gaus_smoothed_SDSS.tolist())
# elif len(desi_flux) > 0:
#     common_ymax = 1.05*max(Gaus_smoothed_DESI.tolist())
# else:
#     common_ymax = 0

plt.figure(figsize=(12,7))
plt.plot(sdss_lamb, sdss_flux, alpha=0.2, color='forestgreen')
plt.plot(sdss_lamb, Gaus_smoothed_SDSS, color='forestgreen')
if SDSS_min <= H_alpha <= SDSS_max:
    plt.axvline(H_alpha, linewidth=2, color='goldenrod', label = u'H\u03B1')
if SDSS_min <= H_beta <= SDSS_max:
    plt.axvline(H_beta, linewidth=2, color='springgreen', label = u'H\u03B2')
if SDSS_min <= Mg2 <= SDSS_max:
    plt.axvline(Mg2, linewidth=2, color='turquoise', label = 'Mg II')
if SDSS_min <= C3_ <= SDSS_max:
    plt.axvline(C3_, linewidth=2, color='indigo', label = 'C III]')
if SDSS_min <= C4 <= SDSS_max:
    plt.axvline(C4, linewidth=2, color='violet', label = 'C IV')
# if SDSS_min <= _O3_ <= SDSS_max:
#     plt.axvline(_O3_, linewidth=2, color='grey', label = '[O III]')
if SDSS_min <= Ly_alpha <= SDSS_max:
    plt.axvline(Ly_alpha, linewidth=2, color='darkviolet', label = u'Ly\u03B1')
if SDSS_min <= Ly_beta <= SDSS_max:
    plt.axvline(Ly_beta, linewidth=2, color='purple', label = u'Ly\u03B2')
# plt.ylim(common_ymin, common_ymax)
plt.xlabel('Wavelength / Å', fontsize = 26)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.ylabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$', fontsize = 26)
plt.title(f'SDSS Spectrum (WISEA J{object_name})', fontsize = 28)
plt.legend(loc = 'best', fontsize = 25)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()