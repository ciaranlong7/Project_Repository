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

object_name = '152517.57+401357.6' #Object A - assigned to me
# object_name = '141923.44-030458.7' #Object B - chosen because of very high redshift
# object_name = '115403.00+003154.0' #Object C - randomly chose a CLAGN, but it had a low redshift also
# object_name = '140957.72-012850.5' #Object D - chosen because of very high z scores
# object_name = '162106.25+371950.7' #Object E - chosen because of very low z scores
# object_name = '135544.25+531805.2' #Object F - chosen because not a CLAGN, but in AGN parent sample & has high z scores
# object_name = '150210.72+522212.2' #Object G - chosen because not a CLAGN, but in AGN parent sample & has low z scores
# object_name = '101536.17+221048.9' #Highly variable AGN object 1 (no SDSS reading in parent sample)
# object_name = '090931.55-011233.3' #Highly variable AGN object 2 (no SDSS reading in parent sample)
# object_name = '151639.06+280520.4' #Object H - chosen because not a CLAGN, but in AGN parent sample & has high z scores & normalised flux change
# object_name = '160833.97+421413.4' #Object I - chosen because not a CLAGN, but in AGN parent sample & has high normalised flux change
# object_name = '164837.68+311652.7' #Object J - chosen because not a CLAGN, but in AGN parent sample & has high z scores
# object_name = '085913.72+323050.8' #Chosen because can't search for SDSS spectrum automatically
# object_name = '115103.77+530140.6' #Object K - chosen to illustrate no need for min dps limit, but need for max gap limit. Norm flux change = 2.19
# object_name = '075448.10+345828.5' #Object L - chosen because only 1 day into ALLWISE-NEOWISE gap
# object_name = '144051.17+024415.8' #Object M - chosen because only 30 days into ALLWISE-NEOWISE gap. Norm flux change = 1.88
# object_name = '164331.90+304835.5' #Object N - chosen due to enourmous Z score
# object_name = '163826.34+382512.1' #Object O - chosen because not a CLAGN, but has enourmous normalised flux change
# object_name = '141535.46+022338.7' #Object P - chosen because of very high z score
# object_name = '121542.99+574702.3' #Object Q - chosen because not a CLAGN, but has a large normalised flux change.
# object_name = '125449.57+574805.3' #Object R - chosen because not a CLAGN, but has a spurious measurement
# object_name = '100523.31+024536.0' #Object S - chosen because has an uncertainty of 0 in its min epoch
# object_name = '114249.08+544709.7' #Object T - chosen because non-CLAGN and has a z score of 141
# object_name = '131630.87+211915.1' #Object U - chosen because non-CLAGN and has a z score of 458
# object_name = '155426.13+200527.7' #chosen because had different z scores
# object_name = '082012.50+352053.8'

#Below are the 3 non-CL AGN that have norm flux difference > threshold.
# object_name = '143054.79+531713.9' #Object V - chosen because non-CLAGN and has a norm flux change of > 1
# object_name = '125449.57+574805.3' #Object R
# object_name = '121947.25+575744.4'

# object_name = '160730.20+560305.5' #Object W - chosen because a CLAGN that exhibits no MIR change over SDSS-DESI range, but does exhibit a change after
# object_name = '115838.31+541619.5' #Object X - chosen because not a CLAGN but shows some variability
# object_name = '213628.50-003811.8' #Object Y - chosen becasue quite different W1 and W2 NFD

# object_name = '111938.02+513315.5' #Highly Variable Non-CL AGN 1

# object_name = '160344.93+450146.4' #outlier = 12 (W2)
# object_name = '112934.69+503103.4' #outlier = 116 (W2)
# object_name = '115634.41+524502.1' #outlier = 240 (W2)
# object_name = '083901.00+232352.4' #outlier = 131 (W2)
# object_name = '161339.24+534552.0' #outlier = 29 (W2)
# object_name = '154713.96+434217.3' #outlier = 35 (W2)
# object_name = '154755.47+424120.0' #outlier = 9.5 (W2)
# object_name = '141440.34+532057.1' #outlier = 27 (W2)
# object_name = '013741.58+205156.6' #outlier = 10.6 (W2)
# object_name = '154050.21+443310.8' #outlier = 27.7 (W2)
# object_name = '115302.69+011027.8' #outlier = 213 (W2)
# object_name = '142334.67+525712.9' #outlier = 4.5 (W2)
# object_name = '153837.03+435132.3' #outlier = 18.5 (W2)
# object_name = '161222.79+543154.9' #outlier = 26 (W2)
# object_name = '142542.91+545710.1' #outlier = 479 (W2)

# object_name = '121449.54+572734.1' #outliers = 163, 140 (W1)
# object_name = '113115.72+533548.9' #outlier = 12.53 (W1)
# object_name = '121234.41+573124.8' #outlier = 239 (W1)

# object_name = '162704.10+420500.0'

#option 1 = Not interested in SDSS or DESI spectrum (MIR only)
#option 2 = Object is a CLAGN, so take SDSS and DESI spectrum from downloads + MIR
#option 3 = download just sdss spectrum from the internet + MIR
#option 4 = download both sdss & desi spectra from the internet + MIR
#option 5 = Object is a CLAGN, so take SDSS and DESI spectrum from downloads (No MIR)
#option 6 = download just sdss spectrum from the internet (No MIR)
#option 7 = download both sdss & desi spectra from the internet (No MIR)
#This prevents unnecessary querying of the databases. DESI database will time out if you spam it.
option = 5

#Selecting which plots you want. Set = 1 if you want that plot
MIR_epoch = 0 #Single epoch plot - set m & n below
MIR_only = 0 #plot with just MIR data on it
MIR_only_no_epoch = 0 #plot with just MIR data on it - not in epochs
SDSS_DESI = 1 #2 plots, each one with just a SDSS or DESI spectrum
SDSS_DESI_comb = 1 #SDSS & DESI spectra on same plot
main_plot = 0 #main plot, with MIR, SDSS & DESI
UV_NFD_plot = 0 #plot with NFD on the top. SDSS & DESI on the bottom
UV_NFD_hist = 0 #histogram of the NFD across each wavelength value

m = 2 # W1 - Change depending on which epoch you wish to look at. m = 0 represents epoch 1. Causes error if (m+1)>number of epochs
n = 2 # W2 - Change depending on which epoch you wish to look at. n = 0 represents epoch 1. Causes error if (n+1)>number of epochs

def flux(mag, k, wavel): # k is the zero magnitude flux density. For W1 & W2, taken from a data table on the search website - https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
    k = (k*(10**(-6))*(c*10**(10)))/(wavel**2) # converting from Jansky to 10-17 ergs/s/cm2/Å. Express c in Angstrom units
    return k*10**(-mag/2.5)

W1_k = 309.540 #Janskys. This means that mag 0 = 309.540 Janskys at the W1 wl.
W2_k = 171.787
W1_wl = 3.4e4 #Angstroms
W2_wl = 4.6e4

def remove_outliers(data, threshold=None):
    """
    Parameters:
    - data: list of tuples [(flux, mjd, unc), ...]
    - threshold: Modified deviation threshold for outlier removal (default=15)

    Returns:
    - list of filtered tuples without outliers
    """
    if not data:
        return data  # Return empty list if input is empty
    
    if my_object == 0:
        threshold = 25
    elif my_object == 1:
        threshold = 9

    flux_values = np.array([entry[0] for entry in data])  # Extract flux values
    median = np.median(flux_values)
    mad = median_abs_deviation(flux_values)

    if mad == 0:  # Avoid division by zero
        print("Warning: MAD is zero, no outliers removed.")
        return data

    modified_deviation = (flux_values - median) / mad
    mask = np.abs(modified_deviation) > threshold  # Identify outliers
    outliers = np.array(data)[mask]  # Extract outlier tuples

    # Print removed outliers
    for outlier, mod_dev in zip(outliers, modified_deviation[mask]):
        print(f"Removing outlier: Flux={outlier[0]}, MJD={outlier[1]}, UNC={outlier[2]} (Modified Deviation = {mod_dev:.2f})")

    return [entry for entry, is_outlier in zip(data, mask) if not is_outlier]

Min_SNR = 3 #Options are 10, 3, or 2. #A (SNR>10), B (3<SNR<10) or C (2<SNR<3)
if Min_SNR == 10: #Select Min_SNR on line above.
    MIR_SNR = 'A'
elif Min_SNR == 3:
    MIR_SNR = 'B'
elif Min_SNR == 2:
    MIR_SNR = 'C'
else:
    print('select a valid min SNR - 10, 3 or 2.')

max_day_gap = 250 #max day gap to linearly interpolate over

Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")
object_names = [x for x in Guo_table4.iloc[:, 0] if pd.notna(x)]
if object_name in object_names:
    my_object = 1 #0 = AGN. 1 = CLAGN
else:
    my_object = 0

if my_object == 1: #If a CLAGN; CLAGN are not in parent sample
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
    parent_sample = pd.read_csv('clean_parent_sample_no_CLAGN.csv') 
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

coord = SkyCoord(SDSS_RA, SDSS_DEC, unit='deg', frame='icrs') #This works

# #Check MJD of a file
# SDSS_file = f'spec-{SDSS_plate}-{SDSS_mjd:.0f}-{SDSS_fiberid}.fits'
# SDSS_file_path = f'clagn_spectra/{SDSS_file}'
# with fits.open(SDSS_file_path) as hdul:
#     header = hdul[0].header
#     print(header)
#     mjd_value = header.get('MJD', 'MJD not found in header')  # Using .get() avoids KeyError if 'MJD' is missing
#     print(f"MJD: {mjd_value}")

def get_sdss_spectra():
    #Automatically querying the SDSS database
    downloaded_SDSS_spec = SDSS.get_spectra_async(plate=SDSS_plate_number, fiberID=SDSS_fiberid_number, mjd=SDSS_mjd)
    if downloaded_SDSS_spec == None:
        downloaded_SDSS_spec = SDSS.get_spectra_async(coordinates=coord, radius=2. * u.arcsec)
        if downloaded_SDSS_spec == None:
            print(f'SDSS Spectrum cannot be found for object_name = {object_name}')
            try:
                SDSS_file = f'spec-{SDSS_plate}-{SDSS_mjd:.0f}-{SDSS_fiberid}.fits'
                SDSS_file_path = f'clagn_spectra/{SDSS_file}'
                with fits.open(SDSS_file_path) as hdul:
                    subset = hdul[1]

                    sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
                    sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms
                    # sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])
                    print('SDSS file is in downloads - will proceed as normal')
                    return sdss_lamb, sdss_flux
            except FileNotFoundError as e:
                print('No SDSS file already downloaded.')
                sdss_flux = []
                sdss_lamb = []
                return sdss_lamb, sdss_flux
        else:
            downloaded_SDSS_spec = downloaded_SDSS_spec[0]
            hdul = HDUList(downloaded_SDSS_spec.get_fits())
            subset = hdul[1]

            sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
            sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms
            # sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])
            return sdss_lamb, sdss_flux
    else:
        downloaded_SDSS_spec = downloaded_SDSS_spec[0]
        hdul = HDUList(downloaded_SDSS_spec.get_fits())
        subset = hdul[1]

        sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
        sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms
        # sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])
        return sdss_lamb, sdss_flux

@retry(stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type((ConnectTimeout, TimeoutError, ConnectionError)))
def get_primary_spectrum(specid): #some objects have multiple spectra for it in DESI- the best one is the 'primary' spectrum
    
    res = client.retrieve_by_specid(specid_list=[specid], include=['specprimary', 'wavelength', 'flux'], dataset_list=['DESI-EDR'])

    records = res.records

    if not records: #no spectrum could be found:
        print(f'DESI Spectrum cannot be found for object_name = {object_name}, DESI specid = {DESI_name}')

        try:
            DESI_file = f'spectrum_desi_{object_name}.csv'
            DESI_file_path = f'clagn_spectra/{DESI_file}'
            DESI_spec = pd.read_csv(DESI_file_path)
            desi_lamb = DESI_spec.iloc[:, 0]  # First column, skipping the first row (header)
            desi_flux = DESI_spec.iloc[:, 1]  # Second column, skipping the first row (header)
            print('DESI file is in downloads - will proceed as normal')
            return desi_lamb, desi_flux
        except FileNotFoundError as e:
            print('No DESI file already downloaded.')
            return [], []

    # Identify the primary spectrum
    spec_primary = np.array([records[jj].specprimary for jj in range(len(records))])

    if not np.any(spec_primary):
        print(f'DESI Spectrum cannot be found for object_name = {object_name}, DESI specid = {DESI_name}')

        try:
            DESI_file = f'spectrum_desi_{object_name}.csv'
            DESI_file_path = f'clagn_spectra/{DESI_file}'
            DESI_spec = pd.read_csv(DESI_file_path)
            desi_lamb = DESI_spec.iloc[:, 0]  # First column
            desi_flux = DESI_spec.iloc[:, 1]  # Second column
            print('DESI file is in downloads - will proceed as normal')
            return desi_lamb, desi_flux
        except FileNotFoundError as e:
            print('No DESI file already downloaded.')
            return [], []

    # Get the index of the primary spectrum
    primary_idx = np.where(spec_primary == True)[0][0]

    # Extract wavelength and flux for the primary spectrum
    desi_lamb = records[primary_idx].wavelength
    desi_flux = records[primary_idx].flux

    return desi_lamb, desi_flux

if option == 1:
    sdss_flux = []
    sdss_lamb = []
    desi_flux = []
    desi_lamb = []
elif option == 2 or option == 5:
    SDSS_file = f'spec-{SDSS_plate}-{SDSS_mjd:.0f}-{SDSS_fiberid}.fits'
    SDSS_file_path = f'clagn_spectra/{SDSS_file}'
    with fits.open(SDSS_file_path) as hdul:
        subset = hdul[1]
        sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
        sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms

    DESI_file = f'spectrum_desi_{object_name}.csv'
    DESI_file_path = f'clagn_spectra/{DESI_file}'
    DESI_spec = pd.read_csv(DESI_file_path)
    desi_lamb = DESI_spec.iloc[:, 0]  # First column
    desi_flux = DESI_spec.iloc[:, 1]  # Second column
elif option == 3 or option == 6:
    desi_flux = []
    desi_lamb = []
    sdss_lamb, sdss_flux = get_sdss_spectra()
elif option == 4 or option == 7:
    client = SparclClient(connect_timeout=10)

    sdss_lamb, sdss_flux = get_sdss_spectra()
    desi_lamb, desi_flux = get_primary_spectrum(int(DESI_name))
else:
    sdss_flux = []
    sdss_lamb = []
    desi_flux = []
    desi_lamb = []
    print('No SDSS or DESI spectrum will be used - select a valid option (1 - 7)')

sfd = sfdmap.SFDMap('SFD_dust_files') #called SFD map, but see - https://github.com/kbarbary/sfdmap/blob/master/README.md
# It explains how "By default, a scaling of 0.86 is applied to the map values to reflect the recalibration by Schlafly & Finkbeiner (2011)"
ebv = sfd.ebv(coord)
print(f"E(B-V): {ebv}")
if 3.1*ebv > 0.53:
    print('Obscured')
else:
    print('Unobscured')

ext_model = G23(Rv=3.1) #Rv=3.1 is typical for MW - Schultz, Wiemer, 1975
# uncorrected_SDSS = sdss_flux
inverse_SDSS_lamb = [1/(x*10**(-4)) for x in sdss_lamb] #need units of inverse microns for extinguishing
inverse_DESI_lamb = [1/(x*10**(-4)) for x in desi_lamb]
sdss_flux = sdss_flux/ext_model.extinguish(inverse_SDSS_lamb, Ebv=ebv) #divide to remove the effect of dust
desi_flux = desi_flux/ext_model.extinguish(inverse_DESI_lamb, Ebv=ebv)

# Correcting for redshift
if object_name in Guo_table4.iloc[:, 0].values:
    object_row = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
    redshift = object_row.iloc[0, 3]
    SDSS_z = redshift
    DESI_z = redshift

sdss_lamb = (sdss_lamb/(1+SDSS_z))
desi_lamb = (desi_lamb/(1+DESI_z))

print(f'Object Name = {object_name}')
print(f'SDSS Redshift = {SDSS_z}')
print(f'DESI Redshift = {DESI_z}')

# # Calculate rolling average manually
# def rolling_average(arr, window_size):
    
#     averages = []
#     for i in range(len(arr) - window_size + 1):
#         avg = np.mean(arr[i:i + window_size])
#         averages.append(avg)
#     return np.array(averages)

#Manual Rolling averages - only uncomment if using (otherwise cuts off first 9 data points)
# SDSS_rolling = rolling_average(sdss_flux, 10)
# DESI_rolling = rolling_average(desi_flux, 10)
# sdss_lamb = sdss_lamb[9:]
# desi_lamb = desi_lamb[9:]
# sdss_flux = sdss_flux[9:]
# desi_flux = desi_flux[9:]

# Gaussian smoothing
# adjust stddev to control the degree of smoothing. Higher stddev means smoother
# https://en.wikipedia.org/wiki/Gaussian_blur
gaussian_kernel = Gaussian1DKernel(stddev=3)

# Smooth the flux data using the Gaussian kernel
if len(sdss_flux) > 0:
    Gaus_smoothed_SDSS = convolve(sdss_flux, gaussian_kernel)
else:
    Gaus_smoothed_SDSS = []
if len(desi_flux) > 0:
    Gaus_smoothed_DESI = convolve(desi_flux, gaussian_kernel)
else:
    Gaus_smoothed_DESI = []
# Gaus_smoothed_SDSS_uncorrected = convolve(uncorrected_SDSS, gaussian_kernel)

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
#Note there are other [O III] lines, such as: 4958.911 A, 4363.210 A
if len(sdss_lamb) > 0:
    SDSS_min = min(sdss_lamb)
    SDSS_max = max(sdss_lamb)
else:
    SDSS_min = 0
    SDSS_max = 1
if len(desi_lamb) > 0:
    DESI_min = min(desi_lamb)
    DESI_max = max(desi_lamb)
else:
    DESI_min = 0
    DESI_max = 1

if SDSS_min < 3000 and SDSS_max > 4020 and DESI_min < 3000 and DESI_max > 4020:

    closest_index_lower_sdss = min(range(len(sdss_lamb)), key=lambda i: abs(sdss_lamb[i] - 3000)) #3000 to avoid Mg2 emission line
    closest_index_upper_sdss = min(range(len(sdss_lamb)), key=lambda i: abs(sdss_lamb[i] - 3920)) #3920 to avoid K Fraunhofer line
    sdss_blue_lamb = sdss_lamb[closest_index_lower_sdss:closest_index_upper_sdss]
    sdss_blue_flux = sdss_flux[closest_index_lower_sdss:closest_index_upper_sdss]

    desi_lamb = desi_lamb.tolist()
    closest_index_lower_desi = min(range(len(desi_lamb)), key=lambda i: abs(desi_lamb[i] - 3000)) #3000 to avoid Mg2 emission line
    closest_index_upper_desi = min(range(len(desi_lamb)), key=lambda i: abs(desi_lamb[i] - 3920)) #3920 to avoid K Fraunhofer line
    desi_blue_lamb = desi_lamb[closest_index_lower_desi:closest_index_upper_desi]
    desi_blue_flux = desi_flux[closest_index_lower_desi:closest_index_upper_desi]

    #interpolating SDSS flux so lambda values match up with DESI . Done this way round because DESI lambda values are closer together.
    sdss_interp_fn = interp1d(sdss_blue_lamb, sdss_blue_flux, kind='linear', fill_value='extrapolate')
    sdss_blue_flux_interp = sdss_interp_fn(desi_blue_lamb) #interpolating the sdss flux to be in line with the desi lambda values

    if np.median(sdss_blue_flux) > np.median(desi_blue_flux): #want high-state minus low-state
        flux_diff = [sdss - desi for sdss, desi in zip(sdss_blue_flux_interp, desi_blue_flux)]
        flux_for_norm = [desi_flux[i] for i in range(len(desi_lamb)) if 3980 <= desi_lamb[i] <= 4020]
        norm_factor = np.median(flux_for_norm)
        UV_NFD = [flux/norm_factor for flux in flux_diff]
    else:
        flux_diff = [desi - sdss for sdss, desi in zip(sdss_blue_flux_interp, desi_blue_flux)]
        flux_for_norm = [sdss_flux[i] for i in range(len(sdss_lamb)) if 3980 <= sdss_lamb[i] <= 4020]
        norm_factor = np.median(flux_for_norm)
        UV_NFD = [flux/norm_factor for flux in flux_diff]
    
    if UV_NFD_plot == 1:
        fig = plt.figure(figsize=(12, 7))
        gs = GridSpec(5, 2, figure=fig)  # 5 rows, 2 columns

        common_ymin = 0
        common_ymax = 1.05*max(Gaus_smoothed_SDSS.tolist()+Gaus_smoothed_DESI.tolist())

        ax1 = fig.add_subplot(gs[0:3, :])
        ax1.plot(desi_blue_lamb, UV_NFD, color = 'darkorange', label = f'{round(DESI_mjd -SDSS_mjd)} days between observations')
        ax1.set_xlabel('Wavelength / Å')
        ax1.set_ylabel('Turned On Flux - Turned Off Flux (Normalised)')
        ax1.set_title(f'UV NFD ({object_name})')

        ax2 = fig.add_subplot(gs[3:, 0])
        ax2.plot(sdss_lamb, sdss_flux, alpha=0.2, color='forestgreen')
        ax2.plot(sdss_lamb, Gaus_smoothed_SDSS, color='forestgreen')
        ax2.set_xlabel('Wavelength / Å')
        ax2.set_ylabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$')
        ax2.set_ylim(common_ymin, common_ymax)
        ax2.set_title('Gaussian Smoothed Plot of SDSS Spectrum')

        ax3 = fig.add_subplot(gs[3:, 1])
        ax3.plot(desi_lamb, desi_flux, alpha=0.2, color='midnightblue')
        ax3.plot(desi_lamb, Gaus_smoothed_DESI, color='midnightblue')
        ax3.set_xlabel('Wavelength / Å')
        ax3.set_ylabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$')
        ax3.set_ylim(common_ymin, common_ymax)
        ax3.set_title('Gaussian Smoothed Plot of DESI Spectrum')

        fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=1.25, wspace=0.2)
        #top and bottom adjust the vertical space on the top and bottom of the figure.
        #left and right adjust the horizontal space on the left and right sides.
        #hspace and wspace adjust the spacing between rows and columns, respectively.
        plt.show()

    if UV_NFD_hist == 1:
        #Histogram of the distribution of flux change values
        median_flux_diff = np.median(UV_NFD)
        MAD_flux_diff = median_abs_deviation(UV_NFD)
        x_start = median_flux_diff - MAD_flux_diff
        x_end = median_flux_diff + MAD_flux_diff
        flux_diff_binsize = (max(UV_NFD)-min(UV_NFD))/50 #50 bins
        bins_flux_diff = np.arange(min(UV_NFD), max(UV_NFD) + flux_diff_binsize, flux_diff_binsize)
        counts, bin_edges = np.histogram(UV_NFD, bins=bins_flux_diff)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        bin_index_start = np.argmin(abs(bin_centers - x_start))
        bin_index_end = np.argmin(abs(bin_centers - x_end))
        height = 1.1*max([counts[bin_index_start], counts[bin_index_end]])

        plt.figure(figsize=(12,7))
        plt.hist(UV_NFD, bins=bins_flux_diff, color='darkorange', edgecolor='black', label=f'binsize = {flux_diff_binsize:.2f}')
        plt.axvline(median_flux_diff, linewidth=2, linestyle='--', color='black', label = f'Median = {median_flux_diff:.2f}')
        plt.plot((x_start, x_end), (height, height), linewidth=2, color='black', label = f'Medain Absolute Deviation = {MAD_flux_diff:.2f}')
        plt.xlabel('Turned On Flux - Turned Off Flux (Normalised)')
        plt.ylabel('Frequency')
        plt.title(f'UV NFD Values Across Wavelength Range ({object_name})')
        plt.legend(loc='upper right')
        plt.show()


# #Plot of SDSS Spectrum - Extinction Corrected vs Uncorrected
# plt.figure(figsize=(12,7))
# plt.plot(sdss_lamb, Gaus_smoothed_SDSS, color = 'blue', label = 'Extinction Corrected')
# plt.plot(sdss_lamb, Gaus_smoothed_SDSS_uncorrected, color = 'blue', label = 'Uncorrected')
# plt.xlabel('Wavelength / Å')
# plt.ylabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$')
# plt.title('SDSS Spectrum - Extinction Corrected vs Uncorrected')
# plt.legend(loc = 'upper right')
# plt.show()

# # #Plot of SDSS Spectrum with uncertainties
# plt.figure(figsize=(12,7))
# plt.errorbar(sdss_lamb, sdss_flux, yerr=sdss_flux_unc, fmt='o', color = 'forestgreen', capsize=5)
# plt.xlabel('Wavelength / Å')
# plt.ylabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$')
# plt.title(f'SDSS Spectrum {object_name}')
# plt.show()


if option >= 1 and option <= 4:
    WISE_query = Irsa.query_region(coordinates=coord, catalog="allwise_p3as_mep", spatial="Cone", radius=2 * u.arcsec)
    NEOWISE_query = Irsa.query_region(coordinates=coord, catalog="neowiser_p1bs_psd", spatial="Cone", radius=2 * u.arcsec)
    WISE_data = WISE_query.to_pandas()
    NEO_data = NEOWISE_query.to_pandas()

    # # # checking out which index corresponds to which column
    # for idx, col in enumerate(WISE_data.columns):
    #     print(f"Index {idx}: {col}")

    WISE_data = WISE_data.sort_values(by=WISE_data.columns[10]) #sort in ascending mjd
    NEO_data = NEO_data.sort_values(by=NEO_data.columns[42]) #sort in ascending mjd

    WISE_data.iloc[:, 6] = pd.to_numeric(WISE_data.iloc[:, 6], errors='coerce')
    filtered_WISE_rows = WISE_data[(WISE_data.iloc[:, 6] == 0) & (WISE_data.iloc[:, 39] == 1) & (WISE_data.iloc[:, 41] == '0000') & (WISE_data.iloc[:, 40] > 5)]
    #filtering for cc_flags == 0 in all bands, qi_fact == 1, no moon masking flag & separation of the WISE instrument to the SAA > 5 degrees. Unlike with Neowise, there is no individual column for cc_flags in each band
    filtered_NEO_rows = NEO_data[(NEO_data.iloc[:, 37] == 1) & (NEO_data.iloc[:, 38] > 5) & (NEO_data.iloc[:, 35] == 0)] #checking for rows where qi_fact == 1 & separation of the WISE instrument to the South Atlantic Anomaly is > 5 degrees & sso_flg ==0
    #"Single-exposure source database entries having qual_frame=0 should be used with extreme caution" - from the column descriptions.
    # The qi_fact column seems to be equal to qual_frame/10.

    #Filtering for good SNR, no cc_flags & no moon scattering flux
    if MIR_SNR == 'C':
        filtered_NEO_rows_W1 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'AB', 'AC', 'AU', 'AX', 'BA', 'BB', 'BC', 'BU', 'BX', 'CA', 'CB', 'CC', 'CU', 'CX'])) & (filtered_NEO_rows.iloc[:, 44] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '01']))]
        filtered_NEO_rows_W2 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'BA', 'CA', 'UA', 'XA', 'AB', 'BB', 'CB', 'UB', 'XB', 'AC', 'BC', 'CC', 'UC', 'XC'])) & (filtered_NEO_rows.iloc[:, 46] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '10']))]
    elif MIR_SNR == 'B':
        filtered_NEO_rows_W1 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'AB', 'AC', 'AU', 'AX', 'BA', 'BB', 'BC', 'BU', 'BX'])) & (filtered_NEO_rows.iloc[:, 44] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '01']))]
        filtered_NEO_rows_W2 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'BA', 'CA', 'UA', 'XA', 'AB', 'BB', 'CB', 'UB', 'XB'])) & (filtered_NEO_rows.iloc[:, 46] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '10']))]
    elif MIR_SNR == 'A':
        filtered_NEO_rows_W1 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'AB', 'AC', 'AU', 'AX'])) & (filtered_NEO_rows.iloc[:, 44] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '01']))]
        filtered_NEO_rows_W2 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'BA', 'CA', 'UA', 'XA'])) & (filtered_NEO_rows.iloc[:, 46] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '10']))]

    mjd_date_W1 = filtered_WISE_rows.iloc[:, 10].tolist() + filtered_NEO_rows_W1.iloc[:, 42].tolist()
    W1_mag = filtered_WISE_rows.iloc[:, 11].tolist() + filtered_NEO_rows_W1.iloc[:, 18].tolist()
    W1_flux = [flux(mag, W1_k, W1_wl) for mag in W1_mag]
    W1_unc = filtered_WISE_rows.iloc[:, 12].tolist() + filtered_NEO_rows_W1.iloc[:, 19].tolist()
    W1_unc = [((unc*np.log(10))/(2.5))*flux for unc, flux in zip(W1_unc, W1_flux)]
    W1_all = list(zip(W1_flux, mjd_date_W1, W1_unc))
    W1_all = [tup for tup in W1_all if not np.isnan(tup[0])] #removing instances where the mag value is NaN

    mjd_date_W2 = filtered_WISE_rows.iloc[:, 10].tolist() + filtered_NEO_rows_W2.iloc[:, 42].tolist()
    W2_mag = filtered_WISE_rows.iloc[:, 14].tolist() + filtered_NEO_rows_W2.iloc[:, 22].tolist()
    W2_flux = [flux(mag, W2_k, W2_wl) for mag in W2_mag]
    W2_unc = filtered_WISE_rows.iloc[:, 15].tolist() + filtered_NEO_rows_W2.iloc[:, 23].tolist()
    W2_unc = [((unc*np.log(10))/(2.5))*flux for unc, flux in zip(W2_unc, W2_flux)]
    W2_all = list(zip(W2_flux, mjd_date_W2, W2_unc))
    W2_all = [tup for tup in W2_all if not np.isnan(tup[0])]

    #removing some outliers
    print(f'W1 data points before filtering = {len(W1_all)}')
    print(f'W2 data points before filtering = {len(W2_all)}')
    W1_all = remove_outliers(W1_all)
    W2_all = remove_outliers(W2_all)

    print(f'W1 data points = {len(W1_all)}')
    print(f'W2 data points = {len(W2_all)}')

    #Object A - The four W1_mag dps with ph_qual C are in rows, 29, 318, 386, 388

    #Below code sorts MIR data.
    #Two assumptions required for code to work:
    #1. The data is sorted in order of oldest mjd to most recent.
    #2. There are 2 or more data points.

    # W1 data first
    W1_list = []
    W1_unc_list = []
    W1_mjds = []
    W1_averages_flux= []
    W1_av_uncs_flux = []
    W1_epoch_dps = []
    W1_av_mjd_date = []
    p = 0
    for i in range(len(W1_all)):
        if i == 0: #first reading - store and move on
            W1_list.append(W1_all[i][0])
            W1_mjds.append(W1_all[i][1])
            W1_unc_list.append(W1_all[i][2])
            continue
        elif i == len(W1_all) - 1: #final data point
            if W1_all[i][1] - W1_all[i-1][1] < 100: #checking if final data point is in the same epoch as previous
                W1_list.append(W1_all[i][0])
                W1_mjds.append(W1_all[i][1])
                W1_unc_list.append(W1_all[i][2])
                W1_averages_flux.append(np.median(W1_list))
                W1_av_mjd_date.append(np.median(W1_mjds))
                #max unc
                mean_unc = (1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list)))
                median_unc = median_abs_deviation(W1_list)
                W1_av_uncs_flux.append(max(mean_unc, median_unc))
                W1_epoch_dps.append(len(W1_list)) #number of data points in this epoch
                if p == m:
                    W1_one_epoch_flux = W1_list
                    W1_one_epoch_uncs_flux = W1_unc_list
                    W1_one_epoch_flux_mjd = W1_mjds
                    mjd_value = W1_all[i][1]
                    p += 1
                p += 1
                continue
            else: #final data point is in an epoch of its own
                W1_averages_flux.append(np.median(W1_list))
                W1_av_mjd_date.append(np.median(W1_mjds))
                #max unc
                mean_unc = (1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list)))
                median_unc = median_abs_deviation(W1_list)
                W1_av_uncs_flux.append(max(mean_unc, median_unc))
                W1_epoch_dps.append(len(W1_list))
                if p == m:
                    W1_one_epoch_flux = W1_list
                    W1_one_epoch_uncs_flux = W1_unc_list
                    W1_one_epoch_flux_mjd = W1_mjds
                    mjd_value = W1_all[i][1]
                p += 1
                W1_epoch_dps.append(1)
                if p == m:
                    W1_one_epoch_flux = [W1_all[i][0]]
                    W1_one_epoch_uncs_flux = [W1_all[i][2]]
                    W1_one_epoch_flux_mjd = [W1_all[i][1]]
                    mjd_value = W1_all[i][1]
                W1_averages_flux.append(W1_all[i][0])
                W1_av_mjd_date.append(W1_all[i][1])
                W1_av_uncs_flux.append(W1_all[i][2])
                continue
        elif W1_all[i][1] - W1_all[i-1][1] < 100: #checking in the same epoch (<100 days between measurements)
            W1_list.append(W1_all[i][0])
            W1_mjds.append(W1_all[i][1])
            W1_unc_list.append(W1_all[i][2])
            continue
        else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
            W1_averages_flux.append(np.median(W1_list))
            W1_av_mjd_date.append(np.median(W1_mjds))
            #max unc
            mean_unc = (1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list)))
            median_unc = median_abs_deviation(W1_list)
            W1_av_uncs_flux.append(max(mean_unc, median_unc))
            W1_epoch_dps.append(len(W1_list))
            if p == m:
                W1_one_epoch_flux = W1_list
                W1_one_epoch_uncs_flux = W1_unc_list
                W1_one_epoch_flux_mjd = W1_mjds
                mjd_value = W1_all[i][1]
                p += 1
            W1_list = []
            W1_mjds = []
            W1_unc_list = []
            W1_list.append(W1_all[i][0])
            W1_mjds.append(W1_all[i][1])
            W1_unc_list.append(W1_all[i][2])
            p += 1
            continue

    # W2 data second
    W2_list = []
    W2_unc_list = []
    W2_mjds = []
    W2_averages_flux= []
    W2_av_uncs_flux = []
    W2_av_mjd_date = []
    W2_epoch_dps = []
    p = 0
    for i in range(len(W2_all)):
        if i == 0: #first reading - store and move on
            W2_list.append(W2_all[i][0])
            W2_mjds.append(W2_all[i][1])
            W2_unc_list.append(W2_all[i][2])
            continue
        elif i == len(W2_all) - 1: #final data point
            if W2_all[i][1] - W2_all[i-1][1] < 100: #checking if final data point is in the same epoch as previous
                W2_list.append(W2_all[i][0])
                W2_mjds.append(W2_all[i][1])
                W2_unc_list.append(W2_all[i][2])
                W2_averages_flux.append(np.median(W2_list))
                W2_av_mjd_date.append(np.median(W2_mjds))
                #max Unc
                mean_unc = (1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list)))
                median_unc = median_abs_deviation(W2_list)
                W2_av_uncs_flux.append(max(mean_unc, median_unc))
                W2_epoch_dps.append(len(W2_list)) #number of data points in this epoch
                if p == n:
                    W2_one_epoch_flux = W2_list
                    W2_one_epoch_uncs_flux = W2_unc_list
                    W2_one_epoch_flux_mjd = W2_mjds
                    mjd_value = W2_all[i][1]
                p += 1
                continue
            else: #final data point is in an epoch of its own
                W2_averages_flux.append(np.median(W2_list))
                W2_av_mjd_date.append(np.median(W2_mjds))
                #max unc
                mean_unc = (1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list)))
                median_unc = median_abs_deviation(W2_list)
                W2_av_uncs_flux.append(max(mean_unc, median_unc))
                W2_epoch_dps.append(len(W2_list))
                if p == n:
                    W2_one_epoch_flux = W2_list
                    W2_one_epoch_uncs_flux = W2_unc_list
                    W2_one_epoch_flux_mjd = W2_mjds
                    mjd_value = W2_all[i][1]
                p += 1
                if p == n:
                    W2_one_epoch_flux = [W2_all[i][0]]
                    W2_one_epoch_uncs_flux = [W2_all[i][2]]
                    W2_one_epoch_flux_mjd = [W2_all[i][1]]
                    mjd_value = W2_all[i][1]
                W2_averages_flux.append(W2_all[i][0])
                W2_av_mjd_date.append(W2_all[i][1])
                W2_av_uncs_flux.append(W2_all[i][2])
                continue
        elif W2_all[i][1] - W2_all[i-1][1] < 100: #checking in the same epoch (<100 days between measurements)
            W2_list.append(W2_all[i][0])
            W2_mjds.append(W2_all[i][1])
            W2_unc_list.append(W2_all[i][2])
            continue
        else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
            W2_averages_flux.append(np.median(W2_list))
            W2_av_mjd_date.append(np.median(W2_mjds))
            #max unc
            mean_unc = (1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list)))
            median_unc = median_abs_deviation(W2_list)
            W2_av_uncs_flux.append(max(mean_unc, median_unc))
            W2_epoch_dps.append(len(W2_list))
            if p == n:
                W2_one_epoch_flux = W2_list
                W2_one_epoch_uncs_flux = W2_unc_list
                W2_one_epoch_flux_mjd = W2_mjds
                mjd_value = W2_all[i][1]
                p += 1
            W2_list = []
            W2_mjds = []
            W2_unc_list = []
            W2_list.append(W2_all[i][0])
            W2_mjds.append(W2_all[i][1])
            W2_unc_list.append(W2_all[i][2])
            p += 1
            continue

    # # Changing mjd date to days since start:
    min_mjd = min([W1_av_mjd_date[0], W2_av_mjd_date[0]])
    min_mjd = 0
    SDSS_mjd = SDSS_mjd - min_mjd
    DESI_mjd = DESI_mjd - min_mjd
    mjd_value = mjd_value - min_mjd
    W1_av_mjd_date = [date - min_mjd for date in W1_av_mjd_date]
    W2_av_mjd_date = [date - min_mjd for date in W2_av_mjd_date]

    # for i in range(len(W1_av_mjd_date)-1):
    #     print(f'{i+1}-{i+2} epoch gap, W1 = {W1_av_mjd_date[i+1]-W1_av_mjd_date[i]}')
    # for j in range(len(W2_av_mjd_date)-1):
    #     print(f'{j+1}-{j+2} epoch gap, W2 = {W2_av_mjd_date[j+1]-W2_av_mjd_date[j]}')

    print(f'Number of MIR W1 epochs = {len(W1_averages_flux)}')
    print(f'Number of MIR W2 epochs = {len(W2_averages_flux)}')

    # # Plotting average raw flux vs mjd since first observation
    # plt.figure(figsize=(12,7))
    # # Flux
    # plt.errorbar(W2_av_mjd_date, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color = 'blue', capsize=5, label = u'W2 (4.6\u03bcm)')
    # plt.errorbar(W1_av_mjd_date, W1_averages_flux, yerr=W1_av_uncs_flux, fmt='o', color = 'blue', capsize=5, label = u'W1 (3.4\u03bcm)')
    # # # Vertical line for SDSS & DESI dates:
    # plt.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--', label = 'SDSS')
    # plt.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--', label = 'DESI')
    # # Labels and Titles
    # plt.xlabel('Days since first observation')
    # # Flux
    # plt.ylabel('Flux / Units of digital numbers')
    # plt.title(f'W1 & W2 Raw Flux vs Time ({object_name})')
    # plt.legend(loc = 'best')
    # plt.show()


    # # Plotting W1 flux Extinction Corrected Vs Uncorrected
    # inverse_W1_lamb = [1/3.4]*len(W1_averages_flux) #need units of inverse microns for extinguishing
    # inverse_W2_lamb = [1/4.6]*len(W2_averages_flux)
    # W1_corrected_flux = W1_averages_flux/ext_model.extinguish(inverse_W1_lamb, Ebv=ebv) #divide to remove the effect of dust
    # W2_corrected_flux = W1_averages_flux/ext_model.extinguish(inverse_W2_lamb, Ebv=ebv)

    # plt.figure(figsize=(12,7))
    # plt.errorbar(W2_av_mjd_date, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color = 'green', capsize=5, label = u'W2 (4.6\u03bcm) Uncorrected')
    # plt.errorbar(W2_av_mjd_date, W2_corrected_flux, yerr=W2_av_uncs_flux, fmt='o', color = 'orange', capsize=5, label = u'W2 (4.6\u03bcm) Corrected')
    # plt.errorbar(W1_av_mjd_date, W1_averages_flux, yerr=W1_av_uncs_flux, fmt='o', color = 'blue', capsize=5, label = u'W1 (3.4\u03bcm) Uncorrected')
    # plt.errorbar(W1_av_mjd_date, W1_corrected_flux, yerr=W1_av_uncs_flux, fmt='o', color = 'blue', capsize=5, label = u'W1 (3.4\u03bcm) Corrected')
    # plt.xlabel('Days since first observation', fontsize = 26)
    # plt.ylabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$', fontsize = 26)
    # plt.xticks(fontsize=26)
    # plt.yticks(fontsize=26)
    # plt.title(f'Flux vs Time (WISEA J{object_name})', fontsize = 28)
    # plt.legend(loc = 'best', fontsize = 25)
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.tight_layout()
    # plt.show()


    # # Plotting colour (W1 mag[average] - W2 mag[average]):
    # colour = [W1 - W2 for W1, W2 in zip(W1_averages_flux, W2_averages_flux)]
    # colour_uncs = [np.sqrt((W1_unc_c)**2+(W2_unc_c)**2) for W1_unc_c, W2_unc_c in zip(W1_av_uncs_flux, W2_av_uncs_flux)]
    # # Uncertainty propagation taken from Hughes & Hase; Z = A - B formula on back cover.

    # plt.figure(figsize=(12,7))
    # plt.errorbar(mjd_date_, colour, yerr=colour_uncs, fmt='o', color = 'orange', capsize=5)
    # #Labels and Titles
    # plt.xlabel('Days since first observation')
    # plt.ylabel('Colour')
    # plt.title('Colour (W1 mag - W2 mag) vs Time')
    # plt.show()


    # Specifically looking at a particular epoch:
    # Change 'm = _' and 'n = _' in above code to change which epoch you look at. m = 0 represents epoch 1.
    # (measurements are taken with a few days hence considered repeats)
    # Create a figure with two subplots (1 row, 2 columns)
    if MIR_epoch == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharex=False)
        # sharex = True explanation:
        # Both subplots will have the same x-axis limits and tick labels.
        # Any changes to the x-axis range (e.g., zooming or setting limits) in one subplot will automatically apply to the other subplot.

        data_point_W1 = list(range(1, len(W1_one_epoch_flux) + 1))
        data_point_W2 = list(range(1, len(W2_one_epoch_flux) + 1))

        # Plot in the first subplot (ax1)
        ax1.errorbar(data_point_W1, W1_one_epoch_flux, yerr=W1_one_epoch_uncs_flux, fmt='o', color='blue', capsize=5, label=u'W1 (3.4\u03bcm)')
        ax1.set_title('W1')
        ax1.set_xlabel('Data Point')
        ax1.set_ylabel('Flux')
        ax1.legend(loc='upper left')

        # Plot in the second subplot (ax2)
        ax2.errorbar(data_point_W2, W2_one_epoch_flux, yerr=W2_one_epoch_uncs_flux, fmt='o', color='orange', capsize=5, label=u'W2 (4.6\u03bcm)')
        ax2.set_title('W2')
        ax2.set_xlabel('Data Point')
        ax2.set_ylabel('Flux')
        ax2.legend(loc='upper left')

        fig.suptitle(f'W1 & W2 band Measurements at Epoch {m+1} and {n+1} respectively - {W1_av_mjd_date[m]:.0f}, {W1_av_mjd_date[n]:.0f} Days Since First Observation respectively', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the main title
        plt.show()

        W2_one_epoch_flux_mjd = [date - min_mjd for date in W2_one_epoch_flux_mjd]
        plt.figure(figsize=(12,7))
        plt.errorbar(W2_one_epoch_flux_mjd, W2_one_epoch_flux, yerr=W2_one_epoch_uncs_flux, fmt='o', color='orange', capsize=5, label=u'W2 (4.6\u03bcm)')
        plt.xlabel('Days Since First Observation', fontsize = 24)
        plt.ylabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$', fontsize = 24)
        plt.title(f'W2 Flux Measurements at Epoch {n+1} (WISEA J{object_name})', fontsize = 24)
        plt.tight_layout()
        plt.show()


    # #Plotting a histogram of a single epoch
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))  # Creates a figure with 1 row and 2 columns

    # bins_W1 = np.arange(min(W1_one_epoch_flux), max(W1_one_epoch_flux) + 0.05, 0.05)
    # ax1.hist(W1_one_epoch_flux, bins=bins_W1, color='blue', edgecolor='black')
    # ax1.set_title('W1')
    # ax1.set_xlabel('Magnitude')
    # ax1.set_ylabel('Frequency')

    # bins_W2 = np.arange(min(W2_one_epoch_flux), max(W2_one_epoch_flux) + 0.05, 0.05)
    # ax2.hist(W2_one_epoch_flux, bins=bins_W2, color='orange', edgecolor='black')
    # ax2.set_title('W2')
    # ax2.set_xlabel('Magnitude')
    # ax2.set_ylabel('Frequency')

    # plt.suptitle(f'W1 & W2 Magnitude Measurements at Epoch {m+1} and {n+1} respectively - {W1_av_mjd_date[m]:.0f} {W1_av_mjd_date[n]:.0f} Days Since First Observation respectively', fontsize=16)
    # plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the main title
    # plt.show()


    # #Plotting a single histogram of a single epoch
    # plt.figure(figsize=(12,7))
    # bins = np.arange(min(W1_one_epoch_flux), max(W1_one_epoch_flux) + 0.01, 0.01)
    # plt.hist(W1_one_epoch_flux, bins=bins, color='blue', edgecolor='black')
    # plt.xlabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$', fontsize = 24)
    # plt.ylabel('Frequency', fontsize = 24)
    # plt.xticks(fontsize=24)
    # plt.yticks(fontsize=24)
    # plt.title(f'W1 Flux Measurements at Epoch {m+1} (WISEA J{object_name})', fontsize = 24)
    # plt.tight_layout()
    # plt.show()


    if MIR_only == 1:
        # Plotting average W1 & W2 mags (or flux) vs days since first observation
        plt.figure(figsize=(12,7))
        # plt.errorbar(W2_av_mjd_date, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', markersize=10, elinewidth=5, color = 'orange', capsize=5, label = u'W2 (4.6\u03bcm)')
        plt.errorbar(W2_av_mjd_date, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color = 'orange', capsize=5, label = u'W2 (4.6\u03bcm)')
        plt.errorbar(W1_av_mjd_date, W1_averages_flux, yerr=W1_av_uncs_flux, fmt='o', color = 'blue', capsize=5, label = u'W1 (3.4\u03bcm)')
        plt.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--', label='SDSS Observation')
        plt.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--', label='DESI Observation')
        plt.xlabel('Days since first observation', fontsize = 26)
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        # plt.ylim(0, 1)
        plt.ylabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$', fontsize = 26)
        plt.title(f'Light Curve (WISEA J{object_name})', fontsize = 28)
        # plt.title(f'AGN Mid-IR Light Curve', fontsize = 28)
        plt.legend(loc = 'best', fontsize = 25)
        plt.tight_layout()
        plt.show()


    if MIR_only_no_epoch == 1:
        # Plotting W1 & W2 flux vs days since first observation. Not binned into epochs
        W2_flux = [tup[0] for tup in W2_all]
        W2_mjd = [tup[1] for tup in W2_all]
        W2_unc = [tup[2] for tup in W2_all]
        W1_flux = [tup[0] for tup in W1_all]
        W1_mjd = [tup[1] for tup in W1_all]
        W1_unc = [tup[2] for tup in W1_all]
        plt.figure(figsize=(12,7))
        plt.errorbar(W2_mjd, W2_flux, yerr=W2_unc, fmt='o', color = 'orange', capsize=5, label = u'W2 (4.6\u03bcm)')
        plt.errorbar(W1_mjd, W1_flux, yerr=W1_unc, fmt='o', color = 'blue', capsize=5, label = u'W1 (3.4\u03bcm)')
        # plt.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--', label='SDSS Observation')
        # plt.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--', label='DESI Observation')
        plt.xlabel('Days since first observation', fontsize = 26)
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.ylabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$', fontsize = 26)
        plt.title(f'Light Curve (WISEA J{object_name})', fontsize = 28)
        plt.legend(loc = 'best', fontsize = 25)
        plt.tight_layout()
        plt.show()


if SDSS_DESI == 1:
    # Plotting Individual SDSS & DESI Spectra individually
    common_ymin = 0
    if len(sdss_flux) > 0 and len(desi_flux) > 0:
        common_ymax = 1.05*max(Gaus_smoothed_SDSS.tolist()+Gaus_smoothed_DESI.tolist())
    elif len(sdss_flux) > 0:
        common_ymax = 1.05*max(Gaus_smoothed_SDSS.tolist())
    elif len(desi_flux) > 0:
        common_ymax = 1.05*max(Gaus_smoothed_DESI.tolist())
    else:
        common_ymax = 0

    plt.figure(figsize=(12,7))
    # plt.plot(sdss_lamb, sdss_flux, alpha=0.2, color='forestgreen')
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
    if SDSS_min <= _O3_ <= SDSS_max:
        plt.axvline(_O3_, linewidth=2, linestyle=':', color='grey', label = '[O III]')
    if SDSS_min <= Ly_alpha <= SDSS_max:
        plt.axvline(Ly_alpha, linewidth=2, color='darkviolet', label = u'Ly\u03B1')
    if SDSS_min <= Ly_beta <= SDSS_max:
        plt.axvline(Ly_beta, linewidth=2, color='purple', label = u'Ly\u03B2')
    plt.ylim(common_ymin, common_ymax)
    plt.xlabel('Wavelength / Å', fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$', fontsize = 26)
    # plt.title(f'SDSS Spectrum (WISEA J{object_name})', fontsize = 28)
    plt.title(f'AGN Emission Spectrum (SDSS)', fontsize = 28)
    plt.legend(loc = 'best', fontsize = 25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,7))
    plt.plot(desi_lamb, desi_flux, alpha=0.2, color='midnightblue')
    plt.plot(desi_lamb, Gaus_smoothed_DESI, color='midnightblue')
    if DESI_min <= H_alpha <= DESI_max:
        plt.axvline(H_alpha, linewidth=2, color='goldenrod', label = u'H\u03B1')
    if DESI_min <= H_beta <= DESI_max:
        plt.axvline(H_beta, linewidth=2, color='springgreen', label = u'H\u03B2')
    if DESI_min <= Mg2 <= DESI_max:
        plt.axvline(Mg2, linewidth=2, color='turquoise', label = 'Mg II')
    if DESI_min <= C3_ <= DESI_max:
        plt.axvline(C3_, linewidth=2, color='indigo', label = 'C III]')
    if DESI_min <= C4 <= DESI_max:
        plt.axvline(C4, linewidth=2, color='violet', label = 'C IV')
    # if DESI_min <= _O3_ <= DESI_max:
    #     plt.axvline(_O3_, linewidth=2, linestyle=':', color='grey', label = '[O III]')
    if DESI_min <= Ly_alpha <= DESI_max:
        plt.axvline(Ly_alpha, linewidth=2, color='darkviolet', label = u'Ly\u03B1')
    if DESI_min <= Ly_beta <= DESI_max:
        plt.axvline(Ly_beta, linewidth=2, color='purple', label = u'Ly\u03B2')
    plt.ylim(common_ymin, common_ymax)
    plt.xlabel('Wavelength / Å', fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$', fontsize = 26)
    plt.title(f'DESI Spectrum (WISEA J{object_name})', fontsize = 28)
    plt.legend(loc = 'best', fontsize = 25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if SDSS_DESI_comb == 1:
    # Plotting SDSS & DESI Spectra on same plot
    common_ymin = 0
    if len(sdss_flux) > 0 and len(desi_flux) > 0:
        common_ymax = 1.05*max(Gaus_smoothed_SDSS.tolist()+Gaus_smoothed_DESI.tolist())
    elif len(sdss_flux) > 0:
        common_ymax = 1.05*max(Gaus_smoothed_SDSS.tolist())
    elif len(desi_flux) > 0:
        common_ymax = 1.05*max(Gaus_smoothed_DESI.tolist())
    else:
        common_ymax = 0

    plt.figure(figsize=(12,7))
    # plt.plot(sdss_lamb, sdss_flux, alpha=0.2, color='forestgreen')
    plt.plot(sdss_lamb, Gaus_smoothed_SDSS, color='forestgreen')
    # plt.plot(desi_lamb, desi_flux, alpha=0.2, color='midnightblue')
    plt.plot(desi_lamb, Gaus_smoothed_DESI, color='midnightblue')
    if SDSS_min <= H_alpha <= SDSS_max or DESI_min <= H_alpha <= DESI_max:
        plt.axvline(H_alpha, linewidth=2, color='goldenrod', label = u'H\u03B1')
    if SDSS_min <= H_beta <= SDSS_max or DESI_min <= H_beta <= DESI_max:
        plt.axvline(H_beta, linewidth=2, color='springgreen', label = u'H\u03B2')
    if SDSS_min <= Mg2 <= SDSS_max or DESI_min <= Mg2 <= DESI_max:
        plt.axvline(Mg2, linewidth=2, color='turquoise', label = 'Mg II')
    if SDSS_min <= C3_ <= SDSS_max or DESI_min <= C3_ <= DESI_max:
        plt.axvline(C3_, linewidth=2, color='indigo', label = 'C III]')
    if SDSS_min <= C4 <= SDSS_max or DESI_min <= C4 <= DESI_max:
        plt.axvline(C4, linewidth=2, color='violet', label = 'C IV')
    if SDSS_min <= _O3_ <= SDSS_max or DESI_min <= _O3_ <= DESI_max:
        plt.axvline(_O3_, linewidth=2, linestyle=':', color='grey', label = '[O III]')
    if SDSS_min <= Ly_alpha <= SDSS_max or DESI_min <= Ly_alpha <= DESI_max:
        plt.axvline(Ly_alpha, linewidth=2, color='darkviolet', label = u'Ly\u03B1')
    if SDSS_min <= Ly_beta <= SDSS_max or DESI_min <= Ly_beta <= DESI_max:
        plt.axvline(Ly_beta, linewidth=2, color='purple', label = u'Ly\u03B2')
    plt.ylim(common_ymin, common_ymax)
    plt.xlabel('Wavelength / Å', fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$', fontsize = 26)
    plt.title(f'SDSS & DESI Spectra {DESI_mjd-SDSS_mjd:.0f} Days Apart', fontsize = 28)
    # plt.title(f'SDSS & DESI Spectra ({object_name})', fontsize = 28)
    plt.legend(loc = 'best', fontsize = 25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if main_plot == 1:
    # Making a big figure with flux & SDSS, DESI spectra added in
    fig = plt.figure(figsize=(12, 7)) # (width, height)
    gs = GridSpec(5, 2, figure=fig)  # 5 rows, 2 columns

    common_ymin = 0
    if len(sdss_flux) > 0 and len(desi_flux) > 0:
        common_ymax = 1.05*max(Gaus_smoothed_SDSS.tolist()+Gaus_smoothed_DESI.tolist())
    elif len(sdss_flux) > 0:
        common_ymax = 1.05*max(Gaus_smoothed_SDSS.tolist())
    elif len(desi_flux) > 0:
        common_ymax = 1.05*max(Gaus_smoothed_DESI.tolist())
    else:
        common_ymax = 0

    # Top plot spanning two columns and three rows (ax1)
    ax1 = fig.add_subplot(gs[0:3, :])  # Rows 0 to 2, both columns
    ax1.errorbar(W2_av_mjd_date, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color='orange', capsize=5, label=u'W2 (4.6\u03bcm)')
    ax1.errorbar(W1_av_mjd_date, W1_averages_flux, yerr=W1_av_uncs_flux, fmt='o', color='blue', capsize=5, label=u'W1 (3.4\u03bcm)')
    ax1.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--', label='SDSS Observation')
    ax1.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--', label='DESI Observation')
    ax1.set_xlabel('Days since first observation', fontsize = 16)
    ax1.set_ylabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$', fontsize = 16, loc='center')
    ax1.tick_params(axis='both', which='major', labelsize = 16)
    ax1.set_title(f'Light Curve (WISEA J{object_name})', fontsize = 22)
    ax1.legend(loc='upper center', fontsize = 18)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Bottom left plot spanning 2 rows and 1 column (ax2)
    ax2 = fig.add_subplot(gs[3:, 0])  # Rows 3 to 4, first column
    ax2.plot(sdss_lamb, sdss_flux, alpha=0.2, color='forestgreen')
    ax2.plot(sdss_lamb, Gaus_smoothed_SDSS, color='forestgreen')
    if SDSS_min <= H_alpha <= SDSS_max:
        ax2.axvline(H_alpha, linewidth=2, color='goldenrod', label = u'H\u03B1')
    if SDSS_min <= H_beta <= SDSS_max:
        ax2.axvline(H_beta, linewidth=2, color='springgreen', label = u'H\u03B2')
    if SDSS_min <= Mg2 <= SDSS_max:
        ax2.axvline(Mg2, linewidth=2, color='turquoise', label = 'Mg II')
    if SDSS_min <= C3_ <= SDSS_max:
        ax2.axvline(C3_, linewidth=2, color='indigo', label = 'C III]')
    if SDSS_min <= C4 <= SDSS_max:
        ax2.axvline(C4, linewidth=2, color='violet', label = 'C IV')
    # if SDSS_min <= _O3_ <= SDSS_max:
    #     ax2.axvline(_O3_, linewidth=2, linestyle=':', color='grey', label = '[O III]')
    if SDSS_min <= Ly_alpha <= SDSS_max:
        ax2.axvline(Ly_alpha, linewidth=2, color='darkviolet', label = u'Ly\u03B1')
    if SDSS_min <= Ly_beta <= SDSS_max:
        ax2.axvline(Ly_beta, linewidth=2, color='purple', label = u'Ly\u03B2')
    ax2.set_xlabel('Wavelength / Å', fontsize = 16)
    ax2.set_ylim(common_ymin, common_ymax)
    ax2.set_ylabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$', fontsize = 15)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    # ax2.xaxis.set_major_locator(MultipleLocator(750))  # Major ticks every 750 Å
    ax2.set_title('SDSS Spectrum', fontsize = 14)
    ax2.legend(loc='upper right', fontsize = 18)

    # Bottom right plot spanning 2 rows and 1 column (ax3)
    ax3 = fig.add_subplot(gs[3:, 1])  # Rows 3 to 4, second column
    ax3.plot(desi_lamb, desi_flux, alpha=0.2, color='midnightblue')
    ax3.plot(desi_lamb, Gaus_smoothed_DESI, color='midnightblue')
    if DESI_min <= H_alpha <= DESI_max:
        ax3.axvline(H_alpha, linewidth=2, color='goldenrod', label = u'H\u03B1')
    if DESI_min <= H_beta <= DESI_max:
        ax3.axvline(H_beta, linewidth=2, color='springgreen', label = u'H\u03B2')
    if DESI_min <= Mg2 <= DESI_max:
        ax3.axvline(Mg2, linewidth=2, color='turquoise', label = 'Mg II')
    if DESI_min <= C3_ <= DESI_max:
        ax3.axvline(C3_, linewidth=2, color='indigo', label = 'C III]')
    if DESI_min <= C4 <= DESI_max:
        ax3.axvline(C4, linewidth=2, color='violet', label = 'C IV')
    # if DESI_min <= _O3_ <= DESI_max:
    #     ax3.axvline(_O3_, linewidth=2, linestyle=':', color='grey', label = '[O III]')
    if DESI_min <= Ly_alpha <= DESI_max:
        ax3.axvline(Ly_alpha, linewidth=2, color='darkviolet', label = u'Ly\u03B1')
    if DESI_min <= Ly_beta <= DESI_max:
        ax3.axvline(Ly_beta, linewidth=2, color='purple', label = u'Ly\u03B2')
    ax3.set_xlabel('Wavelength / Å', fontsize = 16)
    ax3.set_ylim(common_ymin, common_ymax)
    ax3.set_yticks([])
    ax3.tick_params(axis='x', which='major', labelsize=16)
    # ax3.xaxis.set_major_locator(MultipleLocator(750))  # Major ticks every 750 Å
    ax3.set_title('DESI Spectrum', fontsize = 14)
    ax3.legend(loc='upper right', fontsize = 18)

    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.975, hspace=1.5, wspace=0)
    #top and bottom adjust the vertical space on the top and bottom of the figure.
    #left and right adjust the horizontal space on the left and right sides.
    #hspace and wspace adjust the spacing between rows and columns, respectively.

    plt.show()