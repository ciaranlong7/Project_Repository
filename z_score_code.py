import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import median_abs_deviation
import scipy.optimize
from astropy.io import fits
from astropy import units as u #In Astropy, a Quantity object combines a numerical value (like a 1D array of flux) with a physical unit (like W/m^2, erg/s, etc.)
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa
from astropy.io.fits.hdu.hdulist import HDUList
from astroquery.sdss import SDSS
from sparcl.client import SparclClient
import sfdmap
from dust_extinction.parameter_averages import G23
from requests.exceptions import ConnectTimeout
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

c = 299792458
client = SparclClient(connect_timeout=10)

my_object = 1 #0 = AGN. 1 = CLAGN
my_sample = 1 #set which AGN sample you want
save_figures = 0 #set to 1 to save figures

parent_sample = pd.read_csv('guo23_parent_sample_no_duplicates.csv')
Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")
if my_sample == 1:
    AGN_sample = pd.read_csv("AGN_Sample.csv")
if my_sample == 2:
    AGN_sample = pd.read_csv("AGN_Sample_two.csv")
if my_sample == 3:
    AGN_sample = pd.read_csv("AGN_Sample_three.csv")

if my_object == 0:
    object_names = AGN_sample.iloc[:, 3].tolist()
elif my_object == 1:
    object_names = [object_name for object_name in Guo_table4.iloc[:, 0] if pd.notna(object_name)]

def flux(mag, k, wavel): # k is the zero magnitude flux density. For W1 & W2, taken from a data table on the search website - https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
    k = (k*(10**(-6))*(c*10**(10)))/(wavel**2) # converting from Jansky to 10-17 ergs/s/cm2/Å. Express c in Angstrom units
    return k*10**(-mag/2.5)

W1_k = 309.540 #Janskys. This means that mag 0 = 309.540 Janskys at the W1 wl.
W2_k = 171.787
W1_wl = 3.4e4 #Angstroms
W2_wl = 4.6e4

def remove_outliers_epochs(data, threshold=10):
    flux_values = np.array([entry[0] for entry in data])  # Extract flux values
    median = np.median(flux_values)
    mad = median_abs_deviation(flux_values)

    if mad == 0:
        print("MAD is zero, no outliers can be detected.")
        return data

    modified_deviation = (flux_values - median) / mad
    mask = np.abs(modified_deviation) > threshold  # Identify outliers
    outliers = np.array(data)[mask]  # Extract outlier tuples

    # Print removed outliers
    for outlier, mod_dev in zip(outliers, modified_deviation[mask]):
        print(f"Removing outlier: Flux={outlier[0]}, MJD={outlier[1]}, UNC={outlier[2]} (Modified Deviation = {mod_dev:.2f})")

    return [entry for entry, is_outlier in zip(data, mask) if not is_outlier]

object_names_list = [] #Keeps track of objects that met MIR data requirements to take z score & absolute change

# z_score & absolute change lists
W1_max = []
W1_max_unc = []
W1_min = []
W1_min_unc = []
W1_high_unc = []
W1_low = []
W1_low_unc = []
W1_median_dev = []
W1_abs_change = []
W1_abs_change_unc = []
W1_abs_change_norm = []
W1_abs_change_norm_unc = []
W1_first_mjd = []
W1_last_mjd = []
W1_epochs = []
W1_mean_uncs = []
W1_min_mjd = []
W1_max_mjd = []

W2_max = []
W2_max_unc = []
W2_min = []
W2_min_unc = []
W2_high_unc = []
W2_low = []
W2_low_unc = []
W2_median_dev = []
W2_abs_change = []
W2_abs_change_unc = []
W2_abs_change_norm = []
W2_abs_change_norm_unc = []
W2_first_mjd = []
W2_last_mjd = []
W2_epochs = []
W2_mean_uncs = []
W2_min_mjd = []
W2_max_mjd = []

mean_zscore = []
mean_zscore_unc = []
mean_NFD = []
mean_NFD_unc = []
median_UV_NFD = []
median_UV_NFD_unc = []

SDSS_mjds = []
DESI_mjds = []

Min_SNR = 3 #Options are 10, 3, or 2. #A (SNR>10), B (3<SNR<10) or C (2<SNR<3)
if Min_SNR == 10: #Select Min_SNR on line above.
    MIR_SNR = 'A'
elif Min_SNR == 3:
    MIR_SNR = 'B'
elif Min_SNR == 2:
    MIR_SNR = 'C'
else:
    print('select a valid min SNR - 10, 3 or 2.')

#SDSS spectrum retrieval method
@retry(stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type((ConnectTimeout, TimeoutError, ConnectionError)))
def get_primary_SDSS_spectrum(SDSS_plate_number, SDSS_fiberid_number, SDSS_mjd, coord, SDSS_plate, SDSS_fiberid):
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
                    sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])
                    print('SDSS file is in downloads - will proceed as normal')
                    return sdss_lamb, sdss_flux, sdss_flux_unc
            except FileNotFoundError as e:
                print('No SDSS file already downloaded.')
                sdss_flux = []
                sdss_lamb = []
                sdss_flux_unc = []
                return sdss_lamb, sdss_flux, sdss_flux_unc
        else:
            downloaded_SDSS_spec = downloaded_SDSS_spec[0]
            hdul = HDUList(downloaded_SDSS_spec.get_fits())
            subset = hdul[1]

            sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
            sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms
            sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])
            return sdss_lamb, sdss_flux, sdss_flux_unc
    else:
        downloaded_SDSS_spec = downloaded_SDSS_spec[0]
        hdul = HDUList(downloaded_SDSS_spec.get_fits())
        subset = hdul[1]

        sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
        sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms
        sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])
        header = hdul[0].header
        mjd_value = header.get('MJD', 'MJD not found in header')  # Using .get() avoids KeyError if 'MJD' is missing
        if SDSS_mjd - mjd_value > 2:
            print(f"MJD from file header: {mjd_value}")
            print(f"MJD from my csv: {SDSS_mjd}")
        return sdss_lamb, sdss_flux, sdss_flux_unc
            
#DESI spectrum retrieval method
@retry(stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type((ConnectTimeout, TimeoutError, ConnectionError)))
def get_primary_DESI_spectrum(targetid): #some objects have multiple spectra for it in DESI- the best one is the 'primary' spectrum
    res = client.retrieve_by_specid(specid_list=[targetid], include=['specprimary', 'wavelength', 'flux', 'ivar'], dataset_list=['DESI-EDR'])

    records = res.records

    if not records: #no spectrum could be found:
        print(f'DESI Spectrum cannot be found for object_name = {object_name}, DESI target_id = {DESI_name}')

        try:
            DESI_file = f'spectrum_desi_{object_name}.csv'
            DESI_file_path = f'clagn_spectra/{DESI_file}'
            DESI_spec = pd.read_csv(DESI_file_path)
            desi_lamb = DESI_spec.iloc[:, 0]  # First column, skipping the first row (header)
            desi_flux = DESI_spec.iloc[:, 1]  # Second column, skipping the first row (header)
            desi_flux_ivar = DESI_spec.iloc[:, 2]
            desi_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in desi_flux_ivar])
            print('DESI file is in downloads - will proceed as normal')
            return desi_lamb, desi_flux, desi_flux_unc
        except FileNotFoundError as e:
            print('No DESI file already downloaded.')
            return [], [], []

    # Identify the primary spectrum
    spec_primary = np.array([records[jj].specprimary for jj in range(len(records))])

    if not np.any(spec_primary):
        print(f'DESI Spectrum cannot be found for object_name = {object_name}, DESI target_id = {DESI_name}')

        try:
            DESI_file = f'spectrum_desi_{object_name}.csv'
            DESI_file_path = f'clagn_spectra/{DESI_file}'
            DESI_spec = pd.read_csv(DESI_file_path)
            desi_lamb = DESI_spec.iloc[:, 0]  # First column, skipping the first row (header)
            desi_flux = DESI_spec.iloc[:, 1]  # Second column, skipping the first row (header)
            desi_flux_ivar = DESI_spec.iloc[:, 2]
            desi_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in desi_flux_ivar])
            print('DESI file is in downloads - will proceed as normal')
            return desi_lamb, desi_flux, desi_flux_unc
        except FileNotFoundError as e:
            print('No DESI file already downloaded.')
            return [], [], []

    # Get the index of the primary spectrum
    primary_idx = np.where(spec_primary == True)[0][0]

    # Extract wavelength and flux for the primary spectrum
    desi_lamb = records[primary_idx].wavelength
    desi_flux = records[primary_idx].flux
    desi_flux_ivar = records[primary_idx].ivar
    desi_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in desi_flux_ivar])
    return desi_lamb, desi_flux, desi_flux_unc

sfd = sfdmap.SFDMap('SFD_dust_files') #called SFD map, but see - https://github.com/kbarbary/sfdmap/blob/master/README.md
# It explains how "By default, a scaling of 0.86 is applied to the map values to reflect the recalibration by Schlafly & Finkbeiner (2011)"
ext_model = G23(Rv=3.1) #Rv=3.1 is typical for MW - Schultz, Wiemer, 1975
gaussian_kernel = Gaussian1DKernel(stddev=3)

g = 0
for object_name in object_names:
    print(g)
    print(object_name)
    g += 1
    # For AGN:
    if my_object == 0:
        object_data = AGN_sample[AGN_sample.iloc[:, 3] == object_name]
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
    #For CLAGN:
    elif my_object == 1:
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

    # Automatically querying catalogues
    coord = SkyCoord(SDSS_RA, SDSS_DEC, unit='deg', frame='icrs') #This works.
    WISE_query = Irsa.query_region(coordinates=coord, catalog="allwise_p3as_mep", spatial="Cone", radius=2 * u.arcsec)
    NEOWISE_query = Irsa.query_region(coordinates=coord, catalog="neowiser_p1bs_psd", spatial="Cone", radius=2 * u.arcsec)
    WISE_data = WISE_query.to_pandas()
    NEO_data = NEOWISE_query.to_pandas()

    WISE_data = WISE_data.sort_values(by=WISE_data.columns[10]) #sort in ascending mjd
    NEO_data = NEO_data.sort_values(by=NEO_data.columns[42]) #sort in ascending mjd

    filtered_WISE_rows_W1 = WISE_data[(WISE_data.iloc[:, 6].astype(str).str[0] == '0') & (WISE_data.iloc[:, 7] == 1) & (WISE_data.iloc[:, 39] == 1) & (WISE_data.iloc[:, 41].astype(str).str[0] == '0') &  (WISE_data.iloc[:, 40] > 5)]
    filtered_WISE_rows_W2 = WISE_data[(WISE_data.iloc[:, 6].astype(str).str[1] == '0') & (WISE_data.iloc[:, 7] == 1) & (WISE_data.iloc[:, 39] == 1) & (WISE_data.iloc[:, 41].astype(str).str[1] == '0') &  (WISE_data.iloc[:, 40] > 5)]
    #filtering for cc_flags (idx6) == 0, cat (idx7) == 1, qi_fact (idx39) == 1, no moon masking flag (idx41) & separation of the WISE instrument to the SAA (idx40) > 5 degrees. Unlike with Neowise, there is no individual column for cc_flags in each band
    filtered_NEO_rows = NEO_data[(NEO_data.iloc[:, 37] == 1) & (NEO_data.iloc[:, 38] > 5) & (NEO_data.iloc[:, 35] == 0)] #checking for rows where qi_fact == 1 & separation of the WISE instrument to the South Atlantic Anomaly is > 5 degrees & sso_flg ==0
    #"Single-exposure source database entries having qual_frame=0 should be used with extreme caution" - from the column descriptions.
    # The qi_fact column seems to be equal to qual_frame/10.

    #Filtering for good SNR, no cc_flags & no moon scattering flux
    if MIR_SNR == 'C':
        filtered_NEO_rows_W1 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].str[0].isin(['A', 'B', 'C'])) & (filtered_NEO_rows.iloc[:, 44] == '') & (filtered_NEO_rows.iloc[:, 39].astype(str).str[0] == '0')]
        filtered_NEO_rows_W2 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].str[1].isin(['A', 'B', 'C'])) & (filtered_NEO_rows.iloc[:, 46] == '') & (filtered_NEO_rows.iloc[:, 39].astype(str).str[1] == '0')]
    elif MIR_SNR == 'B':
        filtered_NEO_rows_W1 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].str[0].isin(['A', 'B'])) & (filtered_NEO_rows.iloc[:, 44] == '') & (filtered_NEO_rows.iloc[:, 39].astype(str).str[0] == '0')]
        filtered_NEO_rows_W2 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].str[1].isin(['A', 'B'])) & (filtered_NEO_rows.iloc[:, 46] == '') & (filtered_NEO_rows.iloc[:, 39].astype(str).str[1] == '0')]
    elif MIR_SNR == 'A':
        filtered_NEO_rows_W1 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].astype(str).str[0] == 'A') & (filtered_NEO_rows.iloc[:, 44] == '') & (filtered_NEO_rows.iloc[:, 39].astype(str).str[0] == '0')]
        filtered_NEO_rows_W2 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].astype(str).str[1] == 'A') & (filtered_NEO_rows.iloc[:, 46] == '') & (filtered_NEO_rows.iloc[:, 39].astype(str).str[1] == '0')]

    mjd_date_W1 = filtered_WISE_rows_W1.iloc[:, 10].tolist() + filtered_NEO_rows_W1.iloc[:, 42].tolist()
    W1_mag = filtered_WISE_rows_W1.iloc[:, 11].tolist() + filtered_NEO_rows_W1.iloc[:, 18].tolist()
    W1_flux = [flux(mag, W1_k, W1_wl) for mag in W1_mag]
    W1_unc = filtered_WISE_rows_W1.iloc[:, 12].tolist() + filtered_NEO_rows_W1.iloc[:, 19].tolist()
    W1_unc = [((unc*np.log(10))/(2.5))*flux for unc, flux in zip(W1_unc, W1_flux)]
    W1_all = list(zip(W1_flux, mjd_date_W1, W1_unc))
    W1_all = [tup for tup in W1_all if not np.isnan(tup[0])] #removing instances where the mag value is NaN

    mjd_date_W2 = filtered_WISE_rows_W2.iloc[:, 10].tolist() + filtered_NEO_rows_W2.iloc[:, 42].tolist()
    W2_mag = filtered_WISE_rows_W2.iloc[:, 14].tolist() + filtered_NEO_rows_W2.iloc[:, 22].tolist()
    W2_flux = [flux(mag, W2_k, W2_wl) for mag in W2_mag]
    W2_unc = filtered_WISE_rows_W2.iloc[:, 15].tolist() + filtered_NEO_rows_W2.iloc[:, 23].tolist()
    W2_unc = [((unc*np.log(10))/(2.5))*flux for unc, flux in zip(W2_unc, W2_flux)]
    W2_all = list(zip(W2_flux, mjd_date_W2, W2_unc))
    W2_all = [tup for tup in W2_all if not np.isnan(tup[0])]

    if len(W1_all) < 2 and len(W2_all) < 2: #checking if there is enough data
        print('No W1 & W2 data')
        continue

    #Below code sorts MIR data.
    #Two assumptions required for code to work:
    #1. The data is sorted in order of oldest mjd to most recent.
    #2. There are 2 or more data points.

    # W1 data first
    W1_mean_unc_counter = []
    if len(W1_all) > 1:
        W1_list = []
        W1_unc_list = []
        W1_mjds = []
        W1_data = []
        a = 0
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
                    mean_unc = (1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list)))
                    median_unc = median_abs_deviation(W1_list)
                    if mean_unc > median_unc:
                        a+=1
                    W1_data.append( ( np.median(W1_list), np.median(W1_mjds), max(mean_unc, median_unc) ) )
                    W1_mean_unc_counter.append(a)
                    continue
                else: #final data point is in an epoch of its own
                    mean_unc = (1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list)))
                    median_unc = median_abs_deviation(W1_list)
                    if mean_unc > median_unc:
                        a+=1
                    W1_data.append( ( np.median(W1_list), np.median(W1_mjds), max(mean_unc, median_unc) ) )

                    W1_data.append( ( W1_all[i][0], W1_all[i][1], W1_all[i][2] ) )
                    W1_mean_unc_counter.append(a)
                    continue
            elif W1_all[i][1] - W1_all[i-1][1] < 100: #checking in the same epoch (<100 days between measurements)
                W1_list.append(W1_all[i][0])
                W1_mjds.append(W1_all[i][1])
                W1_unc_list.append(W1_all[i][2])
                continue
            else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                mean_unc = (1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list)))
                median_unc = median_abs_deviation(W1_list)
                if mean_unc > median_unc:
                    a+=1
                W1_data.append( ( np.median(W1_list), np.median(W1_mjds), max(mean_unc, median_unc) ) )

                W1_list = []
                W1_mjds = []
                W1_unc_list = []
                W1_list.append(W1_all[i][0])
                W1_mjds.append(W1_all[i][1])
                W1_unc_list.append(W1_all[i][2])
                continue
        #out of for loop now
    else:
        W1_data = [ (0,0,0) ]
        W1_mean_unc_counter.append(np.nan)

    # W2 data second
    W2_mean_unc_counter = []
    if len(W2_all) > 1:
        W2_list = []
        W2_unc_list = []
        W2_mjds = []
        W2_data = []
        a = 0
        for i in range(len(W2_all)):
            if i == 0: #first reading - store and move on
                W2_list.append(W2_all[i][0])
                W2_mjds.append(W2_all[i][1])
                W2_unc_list.append(W2_all[i][2])
                continue
            elif i == len(W2_all) - 1: #if final data point, close the epoch
                if W2_all[i][1] - W2_all[i-1][1] < 100: #checking if final data point is in the same epoch as previous
                    W2_list.append(W2_all[i][0])
                    W2_mjds.append(W2_all[i][1])
                    W2_unc_list.append(W2_all[i][2])
                    mean_unc = (1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list)))
                    median_unc = median_abs_deviation(W2_list)
                    if mean_unc > median_unc:
                        a+=1
                    W2_data.append( ( np.median(W2_list), np.median(W2_mjds), max(mean_unc, median_unc) ) )
                    W2_mean_unc_counter.append(a)
                    continue
                else: #final data point is in an epoch of its own
                    mean_unc = (1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list)))
                    median_unc = median_abs_deviation(W2_list)
                    if mean_unc > median_unc:
                        a+=1
                    W2_data.append( ( np.median(W2_list), np.median(W2_mjds), max(mean_unc, median_unc) ) )

                    W2_data.append( ( W2_all[i][0], W2_all[i][1], W2_all[i][2] ) )
                    W2_mean_unc_counter.append(a)
                    continue
            elif W2_all[i][1] - W2_all[i-1][1] < 100: #checking in the same epoch (<100 days between measurements)
                W2_list.append(W2_all[i][0])
                W2_mjds.append(W2_all[i][1])
                W2_unc_list.append(W2_all[i][2])
                continue
            else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                mean_unc = (1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list)))
                median_unc = median_abs_deviation(W2_list)
                if mean_unc > median_unc:
                    a+=1
                W2_data.append( ( np.median(W2_list), np.median(W2_mjds), max(mean_unc, median_unc) ) )

                W2_list = []
                W2_mjds = []
                W2_unc_list = []
                W2_list.append(W2_all[i][0])
                W2_mjds.append(W2_all[i][1])
                W2_unc_list.append(W2_all[i][2])
                continue
    else:
        W2_data = [ (0,0,0) ]
        W2_mean_unc_counter.append(np.nan)

    #want a minimum of 9 (out of ~24 possible) epochs to conduct analysis on.
    if len(W1_data) > 8:
        W1_data = remove_outliers_epochs(W1_data)
        if len(W1_data) > 8:
            m = 0
            W1_first = W1_data[0][1]
            W1_last = W1_data[-1][1]
        else:
            m = 1
            W1_first = np.nan
            W1_last = np.nan
    else:
        m = 1
        W1_first = np.nan
        W1_last = np.nan
    if len(W2_data) > 8:
        W2_data = remove_outliers_epochs(W2_data)
        if len(W2_data) > 8:
            n = 0
            W2_first = W2_data[0][1]
            W2_last = W2_data[-1][1]
        else:
            n = 1
            W2_first = np.nan
            W2_last = np.nan
    else:
        n = 1
        W2_first = np.nan
        W2_last = np.nan
    if m == 1 and n == 1:
        print('Not enough epochs in W1 & W2')
        continue

    SDSS_mjd_for_dnl = SDSS_mjd
    SDSS_mjds.append(SDSS_mjd)
    DESI_mjds.append(DESI_mjd)

    if save_figures == 1:
        fig = plt.figure(figsize=(12,7))
    if m == 0 and n == 0:
        min_mjd = min([W1_data[0][1], W2_data[0][1]])
        SDSS_mjd = SDSS_mjd - min_mjd
        DESI_mjd = DESI_mjd - min_mjd
        W1_av_mjd_date = [tup[1] - min_mjd for tup in W1_data]
        W2_av_mjd_date = [tup[1] - min_mjd for tup in W2_data]
        W1_averages_flux = [tup[0] for tup in W1_data]
        W1_av_uncs_flux = [tup[2] for tup in W1_data]
        W2_averages_flux = [tup[0] for tup in W2_data]
        W2_av_uncs_flux = [tup[2] for tup in W2_data]
        if save_figures == 1:
            plt.errorbar(W2_av_mjd_date, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color = 'orange', capsize=5, label = u'W2 (4.6\u03bcm)')
            plt.errorbar(W1_av_mjd_date, W1_averages_flux, yerr=W1_av_uncs_flux, fmt='o', color = 'blue', capsize=5, label = u'W1 (3.4\u03bcm)')
    elif n == 0:
        min_mjd = W2_data[0][1]
        SDSS_mjd = SDSS_mjd - min_mjd
        DESI_mjd = DESI_mjd - min_mjd
        W2_av_mjd_date = [tup[1] - min_mjd for tup in W2_data]
        W2_averages_flux = [tup[0] for tup in W2_data]
        W2_av_uncs_flux = [tup[2] for tup in W2_data]
        if save_figures == 1:
            plt.errorbar(W2_av_mjd_date, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color = 'orange', capsize=5, label = u'W2 (4.6\u03bcm)')
    elif m == 0:
        min_mjd = W1_data[0][1]
        SDSS_mjd = SDSS_mjd - min_mjd
        DESI_mjd = DESI_mjd - min_mjd
        min_mjd = W1_data[0][1]
        W1_averages_flux = [tup[0] for tup in W1_data]
        W1_av_uncs_flux = [tup[2] for tup in W1_data]
        W1_av_uncs_flux = [((tup[2]*np.log(10))/(2.5))*flux for tup, flux in zip(W1_data, W1_averages_flux)]
        if save_figures == 1:
            plt.errorbar(W1_av_mjd_date, W1_averages_flux, yerr=W1_av_uncs_flux, fmt='o', color = 'blue', capsize=5, label = u'W1 (3.4\u03bcm)')

    if save_figures == 1:
        plt.xlabel('Days since first observation', fontsize = 26)
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.ylabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$', fontsize = 26)
        plt.title(f'Flux vs Time (WISEA J{object_name})', fontsize = 28)
        plt.tight_layout()
        if my_object == 0:
            fig.savefig(f'C:/Users/ciara/Dropbox/University/University Work/Fourth Year/Project/AGN Figures - Sample {my_sample}/{object_name} - Flux vs Time.png', dpi=300, bbox_inches='tight')
        elif my_object == 1:
            fig.savefig(f'C:/Users/ciara/Dropbox/University/University Work/Fourth Year/Project/CLAGN Figures/{object_name} - Flux vs Time.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    if my_object == 0: #AGN
        sdss_lamb, sdss_flux, sdss_flux_unc = get_primary_SDSS_spectrum(SDSS_plate_number, SDSS_fiberid_number, SDSS_mjd_for_dnl, coord, SDSS_plate, SDSS_fiberid)
        desi_lamb, desi_flux, desi_flux_unc = get_primary_DESI_spectrum(int(DESI_name))
    elif my_object == 1:
        #SDSS
        try:        
            SDSS_file = f'spec-{SDSS_plate}-{SDSS_mjd_for_dnl:.0f}-{SDSS_fiberid}.fits'
            SDSS_file_path = f'clagn_spectra/{SDSS_file}'
            with fits.open(SDSS_file_path) as hdul:
                subset = hdul[1]
                sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
                sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms
                sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])
        except FileNotFoundError as e:
            print('SDSS File not found - trying download')
            sdss_lamb, sdss_flux, sdss_flux_unc = get_primary_SDSS_spectrum(SDSS_plate_number, SDSS_fiberid_number, SDSS_mjd_for_dnl, coord, SDSS_plate, SDSS_fiberid)
        #DESI
        try:
            DESI_file = f'spectrum_desi_{object_name}.csv'
            DESI_file_path = f'clagn_spectra/{DESI_file}'
            DESI_spec = pd.read_csv(DESI_file_path)
            desi_lamb = DESI_spec.iloc[:, 0]
            desi_flux = DESI_spec.iloc[:, 1]
            desi_flux_ivar = DESI_spec.iloc[:, 2]
            desi_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in desi_flux_ivar])
        except FileNotFoundError as e:
            print('DESI File not found - trying download')
            desi_lamb, desi_flux, desi_flux_unc = get_primary_DESI_spectrum(int(DESI_name))

    ebv = sfd.ebv(coord)
    inverse_SDSS_lamb = [1/(x*10**(-4)) for x in sdss_lamb] #need units of inverse microns for extinguishing
    inverse_DESI_lamb = [1/(x*10**(-4)) for x in desi_lamb]
    sdss_flux = sdss_flux/ext_model.extinguish(inverse_SDSS_lamb, Ebv=ebv) #divide to remove the effect of dust
    sdss_flux_unc = sdss_flux_unc/ext_model.extinguish(inverse_SDSS_lamb, Ebv=ebv)
    desi_flux = desi_flux/ext_model.extinguish(inverse_DESI_lamb, Ebv=ebv)
    desi_flux_unc = desi_flux_unc/ext_model.extinguish(inverse_DESI_lamb, Ebv=ebv)

    sdss_lamb = (sdss_lamb/(1+SDSS_z))
    desi_lamb = (desi_lamb/(1+DESI_z))

    if len(sdss_lamb) > 0 and max(sdss_flux) > 0:
        SDSS_min = min(sdss_lamb)
        SDSS_max = max(sdss_lamb)
    else:
        SDSS_min = 0
        SDSS_max = 1
    if len(desi_lamb) > 0 and max(desi_flux) > 0:
        DESI_min = min(desi_lamb)
        DESI_max = max(desi_lamb)
    else:
        DESI_min = 0
        DESI_max = 1

    #UV analysis
    if SDSS_min < 3000 and SDSS_max > 4020 and DESI_min < 3000 and DESI_max > 4020:
        closest_index_lower_sdss = min(range(len(sdss_lamb)), key=lambda i: abs(sdss_lamb[i] - 3000)) #3000 to avoid Mg2 emission line
        closest_index_upper_sdss = min(range(len(sdss_lamb)), key=lambda i: abs(sdss_lamb[i] - 3920)) #3920 to avoid K Fraunhofer line
        sdss_blue_lamb = sdss_lamb[closest_index_lower_sdss:closest_index_upper_sdss]
        sdss_blue_flux = sdss_flux[closest_index_lower_sdss:closest_index_upper_sdss]
        sdss_blue_flux_unc = sdss_flux_unc[closest_index_lower_sdss:closest_index_upper_sdss]

        desi_lamb = desi_lamb.tolist()
        closest_index_lower_desi = min(range(len(desi_lamb)), key=lambda i: abs(desi_lamb[i] - 3000)) #3000 to avoid Mg2 emission line
        closest_index_upper_desi = min(range(len(desi_lamb)), key=lambda i: abs(desi_lamb[i] - 3920)) #3920 to avoid K Fraunhofer line
        desi_blue_lamb = desi_lamb[closest_index_lower_desi:closest_index_upper_desi]
        desi_blue_flux = desi_flux[closest_index_lower_desi:closest_index_upper_desi]
        desi_blue_flux_unc = desi_flux_unc[closest_index_lower_desi:closest_index_upper_desi]


        #interpolating SDSS flux so lambda values match up with DESI . Done this way round because DESI lambda values are closer together.
        sdss_interp_fn = interp1d(sdss_blue_lamb, sdss_blue_flux, kind='linear', fill_value='extrapolate')
        sdss_blue_flux_interp = sdss_interp_fn(desi_blue_lamb) #interpolating the sdss flux to be in line with the desi lambda values

        # Interpolation function for SDSS uncertainties
        def interpolate_uncertainty(SDSS_wavel, SDSS_flux_unc, DESI_wavel):

            SDSS_wavel = np.array(SDSS_wavel)
            SDSS_flux_unc = np.array(SDSS_flux_unc)
            DESI_wavel = np.array(DESI_wavel)

            interpolated_uncs = np.zeros_like(DESI_wavel)

            for i, desi_w in enumerate(DESI_wavel):
                # Find indices of nearest SDSS points
                before_idx = np.searchsorted(SDSS_wavel, desi_w) - 1
                after_idx = before_idx + 1

                # Handle edge cases
                if before_idx < 0:
                    interpolated_uncs[i] = SDSS_flux_unc[after_idx]  # Use the first available uncertainty
                elif after_idx >= len(SDSS_wavel):
                    interpolated_uncs[i] = SDSS_flux_unc[before_idx]  # Use the last available uncertainty
                else:
                    # Get the SDSS wavelength and uncertainty values
                    x1, x2 = SDSS_wavel[before_idx], SDSS_wavel[after_idx]
                    sigma1, sigma2 = SDSS_flux_unc[before_idx], SDSS_flux_unc[after_idx]

                    # Linear uncertainty interpolation formula
                    weight1 = (x2 - desi_w) / (x2 - x1)
                    weight2 = (desi_w - x1) / (x2 - x1)

                    interpolated_uncs[i] = np.sqrt((weight1*sigma1)**2 + (weight2*sigma2)**2)

            return interpolated_uncs
        
        SDSS_unc_interp = interpolate_uncertainty(sdss_blue_lamb, sdss_blue_flux_unc, desi_blue_lamb)

        flux_difference_unc = np.where(
        np.isnan(desi_blue_flux_unc) & np.isnan(SDSS_unc_interp),  # If both are NaN
        0,  # Set to 0
        np.where(
            np.isnan(desi_blue_flux_unc),  # If only desi_blue_flux_unc is NaN
            SDSS_unc_interp,  
            np.where(
                np.isnan(SDSS_unc_interp),  # If only SDSS_unc_interp is NaN
                desi_blue_flux_unc,
                np.sqrt(SDSS_unc_interp**2 + np.array(desi_blue_flux_unc)**2)))) #otherwise propagate the uncertainty as normal.

        if np.median(sdss_blue_flux) > np.median(desi_blue_flux): #want high-state minus low-state
            flux_diff = [sdss - desi for sdss, desi in zip(sdss_blue_flux_interp, desi_blue_flux)]
            flux_for_norm = [desi_flux[i] for i in range(len(desi_lamb)) if 3980 <= desi_lamb[i] <= 4020]
            norm_factor = np.median(flux_for_norm)
            norm_factor_unc = median_abs_deviation(flux_for_norm)
            UV_NFD = [flux/norm_factor for flux in flux_diff]
            UV_NFD_unc_list = [abs(UV_NFD_value)*np.sqrt((flux_difference_unc_value/flux_diff_value)**2 + (norm_factor_unc/norm_factor)**2)
                            for UV_NFD_value, flux_difference_unc_value, flux_diff_value in zip (UV_NFD, flux_difference_unc, flux_diff)]
        else:
            flux_diff = [desi - sdss for sdss, desi in zip(sdss_blue_flux_interp, desi_blue_flux)]
            flux_for_norm = [sdss_flux[i] for i in range(len(sdss_lamb)) if 3980 <= sdss_lamb[i] <= 4020]
            norm_factor = np.median(flux_for_norm)
            norm_factor_unc = median_abs_deviation(flux_for_norm)
            UV_NFD = [flux/norm_factor for flux in flux_diff]
            UV_NFD_unc_list = [abs(UV_NFD_value)*np.sqrt((flux_difference_unc_value/flux_diff_value)**2 + (norm_factor_unc/norm_factor)**2)
                            for UV_NFD_value, flux_difference_unc_value, flux_diff_value in zip (UV_NFD, flux_difference_unc, flux_diff)]
        
        median_UV_NFD.append(np.median(UV_NFD))

        #Now calculating unc in UV_NFD. Chi-squared fitting y = mx+c to the UV_NFD plot. uncertainty in m is the uncertainty in UV_NFD
        #xval is desi_blue_lamb. y_val is UV_NFD.
        desi_blue_lamb = np.array(desi_blue_lamb)

        def model_funct(x, vals):
            return vals[0] + vals[1]*x
        
        initial = np.array([max(UV_NFD), -0.001]) # Initial guess for fit parameters

        def chisq(modelparams, x_data, y_data, y_err):
            chisqval=0
            for i in range(len(x_data)):
                chisqval += ((y_data[i] - model_funct(x_data[i], modelparams))/y_err[i])**2
            return chisqval
        
        fit = scipy.optimize.minimize(chisq, initial, args=(desi_blue_lamb, UV_NFD, UV_NFD_unc_list), method="L-BFGS-B", jac="2-point")
        print(fit)
        a_soln = fit.x[0]
        b_soln = fit.x[1]

        fit_line = model_funct(desi_blue_lamb, [a_soln, b_soln])

        dist_from_grad = [UV_NFD_val - fit_line_val for UV_NFD_val, fit_line_val in zip(UV_NFD, fit_line)]

        UV_NFD_unc = median_abs_deviation(dist_from_grad)
        median_UV_NFD_unc.append(UV_NFD_unc)
    else:
        median_UV_NFD.append(np.nan)
        median_UV_NFD_unc.append(np.nan)

    #MIR analysis:
    if m == 0: #Good W1 if true
        if n == 0: #Good W2 if true
            #Good W1 & W2
            object_names_list.append(object_name)
            W1_mean_uncs.append(W1_mean_unc_counter[0])
            W2_mean_uncs.append(W2_mean_unc_counter[0])
            W1_epochs.append(len(W1_data))
            W2_epochs.append(len(W2_data))
            
            W1_median_dev.append(median_abs_deviation(W1_averages_flux))

            W2_median_dev.append(median_abs_deviation(W2_averages_flux))

            W1_largest = sorted(W1_averages_flux, reverse=True)[0]
            W1_largest_unc = W1_av_uncs_flux[W1_averages_flux.index(W1_largest)] #NOT the largest unc. This is the unc in the largest flux value
            W1_smallest = sorted(W1_averages_flux)[0]
            W1_smallest_unc = W1_av_uncs_flux[W1_averages_flux.index(W1_smallest)]

            W1_high_unc.append(W1_largest_unc)
            W1_low.append(W1_smallest)
            W1_low_unc.append(W1_smallest_unc)

            #uncertainty in absolute flux change
            W1_abs = abs(W1_largest-W1_smallest)
            W1_abs_unc = np.sqrt(W1_largest_unc**2 + W1_smallest_unc**2)

            #uncertainty in normalised flux change
            W1_abs_norm = ((W1_abs)/(W1_smallest))
            W1_abs_norm_unc = W1_abs_norm*np.sqrt(((W1_abs_unc)/(W1_abs))**2 + ((W1_smallest_unc)/(W1_smallest))**2)

            #uncertainty in z score
            W1_z_score_max = (W1_largest-W1_smallest)/(W1_largest_unc)
            W1_z_score_max_unc = abs(W1_z_score_max*((W1_abs_unc)/(W1_abs)))
            W1_z_score_min = (W1_smallest-W1_largest)/(W1_smallest_unc)
            W1_z_score_min_unc = abs(W1_z_score_min*((W1_abs_unc)/(W1_abs)))

            W1_max.append(W1_z_score_max)
            W1_max_unc.append(W1_z_score_max_unc)
            W1_min.append(W1_z_score_min)
            W1_min_unc.append(W1_z_score_min_unc)
            W1_abs_change.append(W1_abs)
            W1_abs_change_unc.append(W1_abs_unc)
            W1_abs_change_norm.append(W1_abs_norm)
            W1_abs_change_norm_unc.append(W1_abs_norm_unc)

            W1_first_mjd.append(W1_first)
            W1_last_mjd.append(W1_last)
            W1_min_mjd.append(W1_smallest_mjd)
            W1_max_mjd.append(W1_largest_mjd)

            W2_largest = sorted(W2_averages_flux, reverse=True)[0]
            W2_largest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_largest)]
            W2_largest_mjd = W2_av_mjd_date[W2_averages_flux.index(W2_largest)]
            W2_smallest = sorted(W2_averages_flux)[0]
            W2_smallest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_smallest)]
            W2_smallest_mjd = W2_av_mjd_date[W2_averages_flux.index(W2_smallest)]

            W2_high_unc.append(W2_largest_unc)
            W2_low.append(W2_smallest)
            W2_low_unc.append(W2_smallest_unc)

            W2_abs = abs(W2_largest-W2_smallest)
            W2_abs_unc = np.sqrt(W2_largest_unc**2 + W2_smallest_unc**2)

            W2_abs_norm = ((W2_abs)/(W2_smallest))
            W2_abs_norm_unc = W2_abs_norm*np.sqrt(((W2_abs_unc)/(W2_abs))**2 + ((W2_smallest_unc)/(W2_smallest))**2)

            W2_z_score_max = (W2_largest-W2_smallest)/(W2_largest_unc)
            W2_z_score_max_unc = abs(W2_z_score_max*((W2_abs_unc)/(W2_abs)))
            W2_z_score_min = (W2_smallest-W2_largest)/(W2_smallest_unc)
            W2_z_score_min_unc = abs(W2_z_score_min*((W2_abs_unc)/(W2_abs)))

            W2_max.append(W2_z_score_max)
            W2_max_unc.append(W2_z_score_max_unc)
            W2_min.append(W2_z_score_min)
            W2_min_unc.append(W2_z_score_min_unc)
            W2_abs_change.append(W2_abs)
            W2_abs_change_unc.append(W2_abs_unc)
            W2_abs_change_norm.append(W2_abs_norm)
            W2_abs_change_norm_unc.append(W2_abs_norm_unc)

            W2_first_mjd.append(W2_first)
            W2_last_mjd.append(W2_last)
            W2_min_mjd.append(W2_smallest_mjd)
            W2_max_mjd.append(W2_largest_mjd)

            zscores = np.sort([abs(W1_z_score_max), abs(W1_z_score_min), abs(W2_z_score_max), abs(W2_z_score_min)]) #sorts in ascending order, nans at end
            zscore_uncs = np.sort([W1_z_score_max_unc, W1_z_score_min_unc, W2_z_score_max_unc, W2_z_score_min_unc])
            if np.isnan(zscores[0]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be nan - all values are nan
                mean_zscore_unc.append(np.nan)
            elif np.isnan(zscores[1]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be zscores[0] - only non nan value
                mean_zscore_unc.append(abs(zscore_uncs[0]))
            elif np.isnan(zscores[2]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be 1/2(zscores[0]+zscores[1])
                mean_zscore_unc.append((1/2)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2))
            elif np.isnan(zscores[3]) == True:
                mean_zscore.append(np.nanmean(zscores))
                mean_zscore_unc.append((1/3)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2 + zscore_uncs[2]**2))
            else:
                mean_zscore.append(np.nanmean(zscores))
                mean_zscore_unc.append((1/4)*np.sqrt(sum(unc**2 for unc in zscore_uncs)))
            
            norm_f_ch = np.sort([W1_abs_norm, W2_abs_norm])
            norm_f_ch_unc = np.sort([W1_abs_norm_unc, W2_abs_norm_unc])
            if np.isnan(norm_f_ch[0]) == True:
                mean_NFD.append(np.nanmean(norm_f_ch)) #will be nan
                mean_NFD_unc.append(np.nan)
            elif np.isnan(norm_f_ch[1]) == True:
                mean_NFD.append(np.nanmean(norm_f_ch)) #will be norm_f_ch[0]
                mean_NFD_unc.append(abs(norm_f_ch_unc[0]))
            else:
                mean_NFD.append(np.nanmean(norm_f_ch))
                mean_NFD_unc.append((1/2)*np.sqrt(sum(unc**2 for unc in norm_f_ch_unc)))
            
        else: 
            #good W1, bad W2
            object_names_list.append(object_name)
            W1_mean_uncs.append(W1_mean_unc_counter[0])
            W2_mean_uncs.append(W2_mean_unc_counter[0])
            W1_epochs.append(len(W1_data))
            W2_epochs.append(len(W2_data))

            W1_median_dev.append(median_abs_deviation(W1_averages_flux))

            W2_high_unc.append(np.nan)
            W2_low.append(np.nan)
            W2_low_unc.append(np.nan)
            W2_median_dev.append(np.nan)

            W1_largest = sorted(W1_averages_flux, reverse=True)[0]
            W1_largest_unc = W1_av_uncs_flux[W1_averages_flux.index(W1_largest)]
            W1_largest_mjd = W1_av_mjd_date[W1_averages_flux.index(W1_largest)]
            W1_smallest = sorted(W1_averages_flux)[0]
            W1_smallest_unc = W1_av_uncs_flux[W1_averages_flux.index(W1_smallest)]
            W1_smallest_mjd = W1_av_mjd_date[W1_averages_flux.index(W1_smallest)]

            W1_high_unc.append(W1_largest_unc)
            W1_low.append(W1_smallest)
            W1_low_unc.append(W1_smallest_unc)

            W1_abs = abs(W1_largest-W1_smallest)
            W1_abs_unc = np.sqrt(W1_largest_unc**2 + W1_smallest_unc**2)

            W1_abs_norm = ((W1_abs)/(W1_smallest))
            W1_abs_norm_unc = W1_abs_norm*np.sqrt(((W1_abs_unc)/(W1_abs))**2 + ((W1_smallest_unc)/(W1_smallest))**2)

            W1_z_score_max = (W1_largest-W1_smallest)/(W1_largest_unc)
            W1_z_score_max_unc = abs(W1_z_score_max*((W1_abs_unc)/(W1_abs)))
            W1_z_score_min = (W1_smallest-W1_largest)/(W1_smallest_unc)
            W1_z_score_min_unc = abs(W1_z_score_min*((W1_abs_unc)/(W1_abs)))

            W1_max.append(W1_z_score_max)
            W1_max_unc.append(W1_z_score_max_unc)
            W1_min.append(W1_z_score_min)
            W1_min_unc.append(W1_z_score_min_unc)
            W1_abs_change.append(W1_abs)
            W1_abs_change_unc.append(W1_abs_unc)
            W1_abs_change_norm.append(W1_abs_norm)
            W1_abs_change_norm_unc.append(W1_abs_norm_unc)

            W1_first_mjd.append(W1_first)
            W1_last_mjd.append(W1_last)
            W1_min_mjd.append(W1_smallest_mjd)
            W1_max_mjd.append(W1_largest_mjd)

            W2_z_score_max = np.nan
            W2_z_score_max_unc = np.nan
            W2_z_score_min = np.nan
            W2_z_score_min_unc = np.nan

            W2_abs_norm = np.nan
            W2_abs_norm_unc = np.nan

            W2_max.append(W2_z_score_max)
            W2_max_unc.append(W2_z_score_max_unc)
            W2_min.append(W2_z_score_min)
            W2_min_unc.append(W2_z_score_min_unc)
            W2_abs_change.append(np.nan)
            W2_abs_change_unc.append(np.nan)
            W2_abs_change_norm.append(np.nan)
            W2_abs_change_norm_unc.append(np.nan)

            W2_first_mjd.append(np.nan)
            W2_last_mjd.append(np.nan)
            W2_min_mjd.append(np.nan)
            W2_max_mjd.append(np.nan)

            zscores = np.sort([abs(W1_z_score_max), abs(W1_z_score_min), abs(W2_z_score_max), abs(W2_z_score_min)]) #sorts in ascending order, nans at end
            zscore_uncs = np.sort([W1_z_score_max_unc, W1_z_score_min_unc, W2_z_score_max_unc, W2_z_score_min_unc])
            if np.isnan(zscores[0]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be nan - all values are nan
                mean_zscore_unc.append(np.nan)
            elif np.isnan(zscores[1]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be zscores[0] - only non nan value
                mean_zscore_unc.append(abs(zscore_uncs[0]))
            elif np.isnan(zscores[2]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be 1/2(zscores[0]+zscores[1])
                mean_zscore_unc.append((1/2)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2))
            elif np.isnan(zscores[3]) == True:
                mean_zscore.append(np.nanmean(zscores))
                mean_zscore_unc.append((1/3)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2 + zscore_uncs[2]**2))
            else:
                mean_zscore.append(np.nanmean(zscores))
                mean_zscore_unc.append((1/4)*np.sqrt(sum(unc**2 for unc in zscore_uncs)))

            norm_f_ch = np.sort([W1_abs_norm, W2_abs_norm])
            norm_f_ch_unc = np.sort([W1_abs_norm_unc, W2_abs_norm_unc])
            if np.isnan(norm_f_ch[0]) == True:
                mean_NFD.append(np.nanmean(norm_f_ch)) #will be nan
                mean_NFD_unc.append(np.nan)
            elif np.isnan(norm_f_ch[1]) == True:
                mean_NFD.append(np.nanmean(norm_f_ch)) #will be norm_f_ch[0]
                mean_NFD_unc.append(abs(norm_f_ch_unc[0]))
            else:
                mean_NFD.append(np.nanmean(norm_f_ch))
                mean_NFD_unc.append((1/2)*np.sqrt(sum(unc**2 for unc in norm_f_ch_unc)))

    else: #Bad W1
        if n == 0: #Good W2 if true
            #Bad W1, good W2
            object_names_list.append(object_name)
            W1_mean_uncs.append(W1_mean_unc_counter[0])
            W2_mean_uncs.append(W2_mean_unc_counter[0])
            W1_epochs.append(len(W1_data))
            W2_epochs.append(len(W2_data))
            
            W1_high_unc.append(np.nan)
            W1_low.append(np.nan)
            W1_low_unc.append(np.nan)

            W1_median_dev.append(np.nan)

            W2_median_dev.append(median_abs_deviation(W2_averages_flux))

            W1_z_score_max = np.nan
            W1_z_score_max_unc = np.nan
            W1_z_score_min = np.nan
            W1_z_score_min_unc = np.nan

            W1_abs_norm = np.nan
            W1_abs_norm_unc = np.nan

            W1_max.append(W1_z_score_max)
            W1_max_unc.append(W1_z_score_max_unc)
            W1_min.append(W1_z_score_min)
            W1_min_unc.append(W1_z_score_min_unc)
            W1_abs_change.append(np.nan)
            W1_abs_change_unc.append(np.nan)
            W1_abs_change_norm.append(np.nan)
            W1_abs_change_norm_unc.append(np.nan)

            W1_first_mjd.append(np.nan)
            W1_last_mjd.append(np.nan)
            W1_min_mjd.append(np.nan)
            W1_max_mjd.append(np.nan)

            W2_largest = sorted(W2_averages_flux, reverse=True)[0]
            W2_largest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_largest)]
            W2_largest_mjd = W2_av_mjd_date[W2_averages_flux.index(W2_largest)]
            W2_smallest = sorted(W2_averages_flux)[0]
            W2_smallest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_smallest)]
            W2_smallest_mjd = W2_av_mjd_date[W2_averages_flux.index(W2_smallest)]

            W2_high_unc.append(W2_largest_unc)
            W2_low.append(W2_smallest)
            W2_low_unc.append(W2_smallest_unc)

            W2_abs = abs(W2_largest-W2_smallest)
            W2_abs_unc = np.sqrt(W2_largest_unc**2 + W2_smallest_unc**2)

            W2_abs_norm = ((W2_abs)/(W2_smallest))
            W2_abs_norm_unc = W2_abs_norm*np.sqrt(((W2_abs_unc)/(W2_abs))**2 + ((W2_smallest_unc)/(W2_smallest))**2)

            W2_z_score_max = (W2_largest-W2_smallest)/(W2_largest_unc)
            W2_z_score_max_unc = abs(W2_z_score_max*((W2_abs_unc)/(W2_abs)))
            W2_z_score_min = (W2_smallest-W2_largest)/(W2_smallest_unc)
            W2_z_score_min_unc = abs(W2_z_score_min*((W2_abs_unc)/(W2_abs)))

            W2_max.append(W2_z_score_max)
            W2_max_unc.append(W2_z_score_max_unc)
            W2_min.append(W2_z_score_min)
            W2_min_unc.append(W2_z_score_min_unc)
            W2_abs_change.append(W2_abs)
            W2_abs_change_unc.append(W2_abs_unc)
            W2_abs_change_norm.append(W2_abs_norm)
            W2_abs_change_norm_unc.append(W2_abs_norm_unc)

            W2_first_mjd.append(W2_first)
            W2_last_mjd.append(W2_last)
            W2_min_mjd.append(W2_smallest_mjd)
            W2_max_mjd.append(W2_largest_mjd)

            zscores = np.sort([abs(W1_z_score_max), abs(W1_z_score_min), abs(W2_z_score_max), abs(W2_z_score_min)]) #sorts in ascending order, nans at end
            zscore_uncs = np.sort([W1_z_score_max_unc, W1_z_score_min_unc, W2_z_score_max_unc, W2_z_score_min_unc])
            if np.isnan(zscores[0]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be nan - all values are nan
                mean_zscore_unc.append(np.nan)
            elif np.isnan(zscores[1]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be zscores[0] - only non nan value
                mean_zscore_unc.append(abs(zscore_uncs[0]))
            elif np.isnan(zscores[2]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be 1/2(zscores[0]+zscores[1])
                mean_zscore_unc.append((1/2)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2))
            elif np.isnan(zscores[3]) == True:
                mean_zscore.append(np.nanmean(zscores))
                mean_zscore_unc.append((1/3)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2 + zscore_uncs[2]**2))
            else:
                mean_zscore.append(np.nanmean(zscores))
                mean_zscore_unc.append((1/4)*np.sqrt(sum(unc**2 for unc in zscore_uncs)))
            
            norm_f_ch = np.sort([W1_abs_norm, W2_abs_norm])
            norm_f_ch_unc = np.sort([W1_abs_norm_unc, W2_abs_norm_unc])
            if np.isnan(norm_f_ch[0]) == True:
                mean_NFD.append(np.nanmean(norm_f_ch)) #will be nan
                mean_NFD_unc.append(np.nan)
            elif np.isnan(norm_f_ch[1]) == True:
                mean_NFD.append(np.nanmean(norm_f_ch)) #will be norm_f_ch[0]
                mean_NFD_unc.append(abs(norm_f_ch_unc[0]))
            else:
                mean_NFD.append(np.nanmean(norm_f_ch))
                mean_NFD_unc.append((1/2)*np.sqrt(sum(unc**2 for unc in norm_f_ch_unc)))

quantifying_change_data = {
    "Object": object_names_list, #0

    "W1 Z Score using Max Unc": W1_max, #1
    "Uncertainty in W1 Z Score using Max Unc": W1_max_unc, #2
    "W1 Z Score using Min Unc": W1_min, #3
    "Uncertainty in W1 Z Score using Min Unc": W1_min_unc, #4
    "W1 Flux Change": W1_abs_change, #5
    "W1 Flux Change Unc": W1_abs_change_unc, #6
    "W1 NFD": W1_abs_change_norm, #7
    "W1 NFD Unc": W1_abs_change_norm_unc, #8

    "W2 Z Score using Max Unc": W2_max, #9
    "Uncertainty in W2 Z Score using Max Unc": W2_max_unc, #10
    "W2 Z Score using Min Unc": W2_min, #11
    "Uncertainty in W2 Z Score using Min Unc": W2_min_unc, #12
    "W2 Flux Change": W2_abs_change, #13
    "W2 Flux Change Unc": W2_abs_change_unc, #14
    "W2 NFD": W2_abs_change_norm, #15
    "W2 NFD Unc": W2_abs_change_norm_unc, #16

    "Mean Z Score": mean_zscore, #17
    "Mean Z Score Unc": mean_zscore_unc, #18
    "Mean NFD": mean_NFD, #19
    "Mean NFD Unc": mean_NFD_unc, #20
    "Median UV Flux Diff On-Off": median_UV_NFD, #21
    "Median UV Flux Diff On-Off Unc": median_UV_NFD_unc, #22

    "SDSS mjd": SDSS_mjds, #23
    "DESI mjd": DESI_mjds, #24
    "W1 First mjd": W1_first_mjd, #25
    "W1 Last mjd": W1_last_mjd, #26
    "W2 First mjd": W2_first_mjd, #27
    "W2 Last mjd": W2_last_mjd, #28
    "W1 Epochs": W1_epochs, #29
    "W2 Epochs": W2_epochs, #30
    "W1 Min Flux": W1_low, #31
    "W1 Min Flux Unc": W1_low_unc, #32
    "W1 Max Flux Unc": W1_high_unc, #33
    "W2 Min Flux": W2_low, #34
    "W2 Min Flux Unc": W2_low_unc, #35
    "W2 Max Flux Unc": W2_high_unc, #36
    "W1 median_abs_dev of Flux": W1_median_dev, #37
    "W2 median_abs_dev of Flux": W2_median_dev, #38
    "W1 Mean Unc Counter": W1_mean_unc_counter, #39
    "W2 Mean Unc Counter": W2_mean_unc_counter, #40
    "W1 min mjd": W1_min_mjd, #41
    "W1 max mjd": W1_max_mjd, #42
    "W2 min mjd": W2_min_mjd, #43
    "W2 max mjd": W2_max_mjd, #44
}

# Convert the data into a DataFrame
df = pd.DataFrame(quantifying_change_data)

if my_object == 0:
    df.to_csv(f"AGN_Quantifying_Change_Sample_{my_sample}.csv", index=False)
elif my_object == 1:
    df.to_csv(f"CLAGN_Quantifying_Change.csv", index=False)