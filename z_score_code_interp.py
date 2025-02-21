import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import median_abs_deviation
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

my_object = 0 #0 = AGN. 1 = CLAGN
my_sample = 3 #set which AGN sample you want
save_figures = 0
optical_analysis = 1 #set = 1 if you wish to do optical analysis. set = 0 if not

parent_sample = pd.read_csv('guo23_parent_sample_no_duplicates.csv')
Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")
if my_sample == 1:
    AGN_sample = pd.read_csv("AGN_Sample.csv")
if my_sample == 2:
    AGN_sample = pd.read_csv("AGN_Sample_two.csv")
if my_sample == 3:
    AGN_sample = pd.read_csv("AGN_Sample_three.csv")

if my_object == 0:
    object_names = AGN_sample.iloc[:, 3]
elif my_object == 1:
    object_names = [object_name for object_name in Guo_table4.iloc[:, 0] if pd.notna(object_name)]

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

object_names_list = []

W1_SDSS_DESI = []
W1_SDSS_DESI_unc = []
W1_DESI_SDSS = []
W1_DESI_SDSS_unc = []
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

W2_SDSS_DESI = []
W2_SDSS_DESI_unc = []
W2_DESI_SDSS = []
W2_DESI_SDSS_unc = []
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

max_day_gap = 250 #max day gap to linearly interpolate over. 250 because quite often the gap between WISE observations is about 210 days.

def find_closest_indices(x_vals, value):
    t = 0
    ninety_first = 0
    ninety_last = 0
    ninety_before = 0
    ninety_after = 0
    if value <= x_vals[0]: #mjd is before first observation
        if x_vals[0] - value > 90:
            t += 1
            print(f'{x_vals[0] - value} days before 1st observation (probably SDSS)')
            return 0, 0, t, ninety_first, ninety_last, ninety_before, ninety_after
        else:
            ninety_first += 1
            return 0, 0, t, ninety_first, ninety_last, ninety_before, ninety_after
    elif value >= x_vals[-1]: #mjd is after last observation
        if value - x_vals[-1] > 90:
            t += 1
            print(f'{value - x_vals[-1]} days after last observation (probably DESI)')
            return 0, 0, t, ninety_first, ninety_last, ninety_before, ninety_after
        else:
            ninety_last += 1
            return 0, 0, t, ninety_first, ninety_last, ninety_before, ninety_after
    for i in range(len(x_vals) - 1):
        if x_vals[i] <= value <= x_vals[i + 1]:
            before_index = i
            after_index = i + 1
            if x_vals[after_index] - x_vals[before_index] > max_day_gap:
                if x_vals[after_index] - value < 90:
                    ninety_before += 1
                elif value - x_vals[before_index] < 90:
                    ninety_after += 1
                else:
                    t += 1
            return before_index, after_index, t, ninety_first, ninety_last, ninety_before, ninety_after

if optical_analysis == 1:
    client = SparclClient(connect_timeout=10)
            
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
                return sdss_lamb, sdss_flux
        else:
            downloaded_SDSS_spec = downloaded_SDSS_spec[0]
            hdul = HDUList(downloaded_SDSS_spec.get_fits())
            subset = hdul[1]

            sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
            sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms
            header = hdul[0].header
            mjd_value = header.get('MJD', 'MJD not found in header')  # Using .get() avoids KeyError if 'MJD' is missing
            if SDSS_mjd - mjd_value > 2:
                print(f"MJD from file header: {mjd_value}")
                print(f"MJD from my csv: {SDSS_mjd}")
            return sdss_lamb, sdss_flux
                
    #DESI spectrum retrieval method
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type((ConnectTimeout, TimeoutError, ConnectionError)))
    def get_primary_DESI_spectrum(targetid): #some objects have multiple spectra for it in DESI- the best one is the 'primary' spectrum
        res = client.retrieve_by_specid(specid_list=[targetid], include=['specprimary', 'wavelength', 'flux'], dataset_list=['DESI-EDR'])

        records = res.records

        if not records: #no spectrum could be found:
            print(f'DESI Spectrum cannot be found for object_name = {object_name}, DESI target_id = {DESI_name}')

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

        # Identify the primary spectrum
        spec_primary = np.array([records[jj].specprimary for jj in range(len(records))])

        if not np.any(spec_primary):
            print(f'DESI Spectrum cannot be found for object_name = {object_name}, DESI target_id = {DESI_name}')

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
        primary_ii = np.where(spec_primary == True)[0][0]

        # Extract wavelength and flux for the primary spectrum
        desi_lamb = records[primary_ii].wavelength
        desi_flux = records[primary_ii].flux

        return desi_lamb, desi_flux

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
        SDSS_mjd = object_data.iloc[0, 5]
        DESI_mjd = object_data.iloc[0, 11]
        if optical_analysis == 1:
            SDSS_plate_number = object_data.iloc[0, 4]
            SDSS_plate = f'{SDSS_plate_number:04}'
            SDSS_fiberid_number = object_data.iloc[0, 6]
            SDSS_fiberid = f"{SDSS_fiberid_number:04}"
            SDSS_z = object_data.iloc[0, 2]
            DESI_z = object_data.iloc[0, 9]
            DESI_name = object_data.iloc[0, 10]
    #For CLAGN:
    elif my_object == 1:
        object_data = parent_sample[parent_sample.iloc[:, 3] == object_name]
        SDSS_RA = object_data.iloc[0, 0]
        SDSS_DEC = object_data.iloc[0, 1]
        SDSS_mjd = object_data.iloc[0, 5]
        DESI_mjd = object_data.iloc[0, 11]
        if optical_analysis == 1:
            SDSS_plate_number = object_data.iloc[0, 4]
            SDSS_plate = f'{SDSS_plate_number:04}'
            SDSS_fiberid_number = object_data.iloc[0, 6]
            SDSS_fiberid = f"{SDSS_fiberid_number:04}"
            SDSS_z = object_data.iloc[0, 2]
            DESI_z = object_data.iloc[0, 9]
            DESI_name = object_data.iloc[0, 10]

    if SDSS_mjd < 55089: #55179 is 14/12/2009 - the date of the 1st ever WISE observation. Accept data that is within 90 days of this
        print(f'SDSS observation was {55179 - SDSS_mjd} days before 1st ever WISE observation.')

    # Automatically querying catalogues
    coord = SkyCoord(SDSS_RA, SDSS_DEC, unit='deg', frame='icrs') #This works.
    WISE_query = Irsa.query_region(coordinates=coord, catalog="allwise_p3as_mep", spatial="Cone", radius=2 * u.arcsec)
    NEOWISE_query = Irsa.query_region(coordinates=coord, catalog="neowiser_p1bs_psd", spatial="Cone", radius=2 * u.arcsec)
    WISE_data = WISE_query.to_pandas()
    NEO_data = NEOWISE_query.to_pandas()

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
    W1_all = remove_outliers(W1_all)
    W2_all = remove_outliers(W2_all)

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
        m = 0
        W1_first = W1_data[0][1]
        W1_last = W1_data[-1][1]
    else:
        m = 1
        W1_first = np.nan
        W1_last = np.nan
    if len(W2_data) > 8:
        n = 0
        W2_first = W2_data[0][1]
        W2_last = W2_data[-1][1]
    else:
        n = 1
        W2_first = np.nan
        W2_last = np.nan
    if m == 1 and n == 1:
        print('Not enough epochs in W1 & W2')
        continue
    
    if SDSS_mjd + 90 < W1_first and SDSS_mjd + 90 < W2_first:
        print(f'SDSS mjd = {SDSS_mjd};before first MIR observation')
        continue
    if DESI_mjd - 90 > W1_last and DESI_mjd - 90 > W2_last:
        print(f'DESI mjd = {DESI_mjd};after last MIR observation')
        continue
    SDSS_mjd_for_dnl = SDSS_mjd
    DESI_mjd_for_dnl = DESI_mjd
    
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
            plt.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--')
            plt.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--')
            plt.errorbar(W2_av_mjd_date, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color = 'orange', capsize=5, label = u'W2 (4.6\u03bcm)')
            plt.errorbar(W1_av_mjd_date, W1_averages_flux, yerr=W1_av_uncs_flux, fmt='o', color = 'blue', capsize=5, label = u'W1 (3.4\u03bcm)')

        before_SDSS_index_W1, after_SDSS_index_W1, q, ninety_first_SDSS_W1, ninety_last_SDSS_W1, ninety_before_SDSS_W1, ninety_after_SDSS_W1 = find_closest_indices(W1_av_mjd_date, SDSS_mjd)
        before_SDSS_index_W2, after_SDSS_index_W2, w, ninety_first_SDSS_W2, ninety_last_SDSS_W2, ninety_before_SDSS_W2, ninety_after_SDSS_W2 = find_closest_indices(W2_av_mjd_date, SDSS_mjd)
        before_DESI_index_W1, after_DESI_index_W1, e, ninety_first_DESI_W1, ninety_last_DESI_W1, ninety_before_DESI_W1, ninety_after_DESI_W1 = find_closest_indices(W1_av_mjd_date, DESI_mjd)
        before_DESI_index_W2, after_DESI_index_W2, r, ninety_first_DESI_W2, ninety_last_DESI_W2, ninety_before_DESI_W2, ninety_after_DESI_W2 = find_closest_indices(W2_av_mjd_date, DESI_mjd)
    elif n == 0:
        min_mjd = W2_data[0][1]
        SDSS_mjd = SDSS_mjd - min_mjd
        DESI_mjd = DESI_mjd - min_mjd
        W2_av_mjd_date = [tup[1] - min_mjd for tup in W2_data]
        W2_averages_flux = [tup[0] for tup in W2_data]
        W2_av_uncs_flux = [tup[2] for tup in W2_data]
        if save_figures == 1:
            plt.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--')
            plt.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--')
            plt.errorbar(W2_av_mjd_date, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color = 'orange', capsize=5, label = u'W2 (4.6\u03bcm)')
        
        before_SDSS_index_W2, after_SDSS_index_W2, w, ninety_first_SDSS_W2, ninety_last_SDSS_W2, ninety_before_SDSS_W2, ninety_after_SDSS_W2 = find_closest_indices(W2_av_mjd_date, SDSS_mjd)
        before_DESI_index_W2, after_DESI_index_W2, r, ninety_first_DESI_W2, ninety_last_DESI_W2, ninety_before_DESI_W2, ninety_after_DESI_W2 = find_closest_indices(W2_av_mjd_date, DESI_mjd)
        q = 1
        e = 1
    elif m == 0:
        min_mjd = W1_data[0][1]
        SDSS_mjd = SDSS_mjd - min_mjd
        DESI_mjd = DESI_mjd - min_mjd
        W1_av_mjd_date = [tup[1] - min_mjd for tup in W1_data]
        W1_averages_flux = [tup[0] for tup in W1_data]
        W1_av_uncs_flux = [tup[2] for tup in W1_data]
        if save_figures == 1:
            plt.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--')
            plt.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--')
            plt.errorbar(W1_av_mjd_date, W1_averages_flux, yerr=W1_av_uncs_flux, fmt='o', color = 'blue', capsize=5, label = u'W1 (3.4\u03bcm)')

        before_SDSS_index_W1, after_SDSS_index_W1, q, ninety_first_SDSS_W1, ninety_last_SDSS_W1, ninety_before_SDSS_W1, ninety_after_SDSS_W1 = find_closest_indices(W1_av_mjd_date, SDSS_mjd)
        before_DESI_index_W1, after_DESI_index_W1, e, ninety_first_DESI_W1, ninety_last_DESI_W1, ninety_before_DESI_W1, ninety_after_DESI_W1 = find_closest_indices(W1_av_mjd_date, DESI_mjd)
        w = 1
        r = 1
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
    
    if q != 0 or e != 0:
        if w != 0 or r != 0:
            print('SDSS or DESI observation not near MIR data')
            continue
    
    if optical_analysis == 1:
        if my_object == 0: #AGN
            sdss_lamb, sdss_flux = get_primary_SDSS_spectrum(SDSS_plate_number, SDSS_fiberid_number, SDSS_mjd_for_dnl, coord, SDSS_plate, SDSS_fiberid)
            desi_lamb, desi_flux = get_primary_DESI_spectrum(int(DESI_name))
        elif my_object == 1:
            #SDSS
            try:        
                SDSS_file = f'spec-{SDSS_plate}-{SDSS_mjd_for_dnl:.0f}-{SDSS_fiberid}.fits'
                SDSS_file_path = f'clagn_spectra/{SDSS_file}'
                with fits.open(SDSS_file_path) as hdul:
                    subset = hdul[1]
                    sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
                    sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms
            except FileNotFoundError as e:
                print('SDSS File not found - trying download')
                sdss_lamb, sdss_flux = get_primary_SDSS_spectrum(SDSS_plate_number, SDSS_fiberid_number, SDSS_mjd_for_dnl, coord, SDSS_plate, SDSS_fiberid)
            #DESI
            try:
                DESI_file = f'spectrum_desi_{object_name}.csv'
                DESI_file_path = f'clagn_spectra/{DESI_file}'
                DESI_spec = pd.read_csv(DESI_file_path)
                desi_lamb = DESI_spec.iloc[:, 0]
                desi_flux = DESI_spec.iloc[:, 1]
            except FileNotFoundError as e:
                print('DESI File not found - trying download')
                desi_lamb, desi_flux = get_primary_DESI_spectrum(int(DESI_name))

        ebv = sfd.ebv(coord)
        inverse_SDSS_lamb = [1/(x*10**(-4)) for x in sdss_lamb] #need units of inverse microns for extinguishing
        inverse_DESI_lamb = [1/(x*10**(-4)) for x in desi_lamb]
        sdss_flux = sdss_flux/ext_model.extinguish(inverse_SDSS_lamb, Ebv=ebv) #divide to remove the effect of dust
        desi_flux = desi_flux/ext_model.extinguish(inverse_DESI_lamb, Ebv=ebv)

        sdss_lamb = (sdss_lamb/(1+SDSS_z))
        desi_lamb = (desi_lamb/(1+DESI_z))
        if len(sdss_flux) > 0:
            Gaus_smoothed_SDSS = convolve(sdss_flux, gaussian_kernel)
        else:
            Gaus_smoothed_SDSS = []
        if len(desi_flux) > 0:
            Gaus_smoothed_DESI = convolve(desi_flux, gaussian_kernel)
        else:
            Gaus_smoothed_DESI = []
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

        #UV analysis
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

            median_UV_NFD.append(np.median(UV_NFD))
            median_UV_NFD_unc.append(median_abs_deviation(UV_NFD))
        else:
            median_UV_NFD.append(np.nan)
            median_UV_NFD_unc.append(np.nan)
    else:
        median_UV_NFD.append(np.nan)
        median_UV_NFD_unc.append(np.nan)

    SDSS_mjds.append(SDSS_mjd_for_dnl)
    DESI_mjds.append(DESI_mjd_for_dnl)
    if q == 0 and e == 0: #Good W1 if true
        if w == 0 and r == 0: #Good W2 if true
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

            if ninety_first_SDSS_W1 == 1: #object is < 90 days before 1st obsv. Let's take 1st MIR obsv as our SDSS_interp.
                W1_SDSS_interp = W1_averages_flux[0]
                W1_SDSS_unc_interp = W1_av_uncs_flux[0]
            elif ninety_last_SDSS_W1 == 1: #object is < 90 days after 1st obsv. Let's take last MIR obsv as our SDSS_interp.
                W1_SDSS_interp = W1_averages_flux[-1]
                W1_SDSS_unc_interp = W1_av_uncs_flux[-1]
            elif ninety_before_SDSS_W1 == 1: #observation is during time of max day gap, but max day gap for interp exceeded.
                                            # however, the SDSS observation is < 90 days before the next observation -
                                            # so approximate SDSS_interp to the next observation.
                                            #For example. MIR epoch 10 MJD = 56000. SDSS MJD = 56500. MIR Epoch 11 MJD = 56501.
                                            # Hence approximate SDSS_interp to MIR epoch 11 - the 'after' index.
                W1_SDSS_interp = W1_averages_flux[after_SDSS_index_W1]
                W1_SDSS_unc_interp = W1_av_uncs_flux[after_SDSS_index_W1]
            elif ninety_after_SDSS_W1 == 1:
                W1_SDSS_interp = W1_averages_flux[before_SDSS_index_W1]
                W1_SDSS_unc_interp = W1_av_uncs_flux[before_SDSS_index_W1]
            else:
                #Linearly interpolating to get interpolated flux on a value in between the data points adjacent to SDSS.
                W1_SDSS_interp = np.interp(SDSS_mjd, W1_av_mjd_date, W1_averages_flux)
                W1_SDSS_unc_interp = np.sqrt((((W1_av_mjd_date[after_SDSS_index_W1] - SDSS_mjd)/(W1_av_mjd_date[after_SDSS_index_W1] - W1_av_mjd_date[before_SDSS_index_W1]))*W1_av_uncs_flux[before_SDSS_index_W1])**2 + (((SDSS_mjd - W1_av_mjd_date[before_SDSS_index_W1])/(W1_av_mjd_date[after_SDSS_index_W1] - W1_av_mjd_date[before_SDSS_index_W1]))*W1_av_uncs_flux[after_SDSS_index_W1])**2)
            
            if ninety_first_DESI_W1 == 1:
                W1_DESI_interp = W1_averages_flux[0]
                W1_DESI_unc_interp = W1_av_uncs_flux[0]
            elif ninety_last_DESI_W1 == 1:
                W1_DESI_interp = W1_averages_flux[-1]
                W1_DESI_unc_interp = W1_av_uncs_flux[-1]
            elif ninety_before_DESI_W1 == 1:
                W1_DESI_interp = W1_averages_flux[after_DESI_index_W1]
                W1_DESI_unc_interp = W1_av_uncs_flux[after_DESI_index_W1]
            elif ninety_after_DESI_W1 == 1:
                W1_DESI_interp = W1_averages_flux[before_DESI_index_W1]
                W1_DESI_unc_interp = W1_av_uncs_flux[before_DESI_index_W1]
            else:
                #Linearly interpolating to get interpolated flux on a value in between the data points adjacent to DESI.
                W1_DESI_interp = np.interp(DESI_mjd, W1_av_mjd_date, W1_averages_flux)
                W1_DESI_unc_interp = np.sqrt((((W1_av_mjd_date[after_DESI_index_W1] - DESI_mjd)/(W1_av_mjd_date[after_DESI_index_W1] - W1_av_mjd_date[before_DESI_index_W1]))*W1_av_uncs_flux[before_DESI_index_W1])**2 + (((DESI_mjd - W1_av_mjd_date[before_DESI_index_W1])/(W1_av_mjd_date[after_DESI_index_W1] - W1_av_mjd_date[before_DESI_index_W1]))*W1_av_uncs_flux[after_DESI_index_W1])**2)
            
            #uncertainty in absolute flux change
            W1_abs = abs(W1_SDSS_interp-W1_DESI_interp)
            W1_abs_unc = np.sqrt(W1_SDSS_unc_interp**2 + W1_DESI_unc_interp**2)

            #uncertainty in normalised flux change
            W1_abs_norm = ((W1_abs)/(W1_smallest))
            W1_abs_norm_unc = W1_abs_norm*np.sqrt(((W1_abs_unc)/(W1_abs))**2 + ((W1_smallest_unc)/(W1_smallest))**2)

            #uncertainty in z score
            W1_z_score_SDSS_DESI = (W1_SDSS_interp-W1_DESI_interp)/(W1_DESI_unc_interp)
            W1_z_score_SDSS_DESI_unc = abs(W1_z_score_SDSS_DESI*((W1_abs_unc)/(W1_abs)))
            W1_z_score_DESI_SDSS = (W1_DESI_interp-W1_SDSS_interp)/(W1_SDSS_unc_interp)
            W1_z_score_DESI_SDSS_unc = abs(W1_z_score_DESI_SDSS*((W1_abs_unc)/(W1_abs)))

            W1_SDSS_DESI.append(W1_z_score_SDSS_DESI)
            W1_SDSS_DESI_unc.append(W1_z_score_SDSS_DESI_unc)
            W1_DESI_SDSS.append(W1_z_score_DESI_SDSS)
            W1_DESI_SDSS_unc.append(W1_z_score_DESI_SDSS_unc)
            W1_abs_change.append(W1_abs)
            W1_abs_change_unc.append(W1_abs_unc)
            W1_abs_change_norm.append(W1_abs_norm)
            W1_abs_change_norm_unc.append(W1_abs_norm_unc)

            W1_first_mjd.append(W1_first)
            W1_last_mjd.append(W1_last)

            W2_largest = sorted(W2_averages_flux, reverse=True)[0]
            W2_largest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_largest)]
            W2_smallest = sorted(W2_averages_flux)[0]
            W2_smallest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_smallest)]

            W2_high_unc.append(W2_largest_unc)
            W2_low.append(W2_smallest)
            W2_low_unc.append(W2_smallest_unc)

            if ninety_first_SDSS_W2 == 1:
                W2_SDSS_interp = W2_averages_flux[0]
                W2_SDSS_unc_interp = W2_av_uncs_flux[0]
            elif ninety_last_SDSS_W2 == 1:
                W2_SDSS_interp = W2_averages_flux[-1]
                W2_SDSS_unc_interp = W2_av_uncs_flux[-1]
            elif ninety_before_SDSS_W2 == 1:
                W2_SDSS_interp = W2_averages_flux[after_SDSS_index_W2]
                W2_SDSS_unc_interp = W2_av_uncs_flux[after_SDSS_index_W2]
            elif ninety_after_SDSS_W2 == 1:
                W2_SDSS_interp = W2_averages_flux[before_SDSS_index_W2]
                W2_SDSS_unc_interp = W2_av_uncs_flux[before_SDSS_index_W2]
            else:
                #Linearly interpolating to get interpolated flux on a value in between the data points adjacent to SDSS.
                W2_SDSS_interp = np.interp(SDSS_mjd, W2_av_mjd_date, W2_averages_flux)
                W2_SDSS_unc_interp = np.sqrt((((W2_av_mjd_date[after_SDSS_index_W2] - SDSS_mjd)/(W2_av_mjd_date[after_SDSS_index_W2] - W2_av_mjd_date[before_SDSS_index_W2]))*W2_av_uncs_flux[before_SDSS_index_W2])**2 + (((SDSS_mjd - W2_av_mjd_date[before_SDSS_index_W2])/(W2_av_mjd_date[after_SDSS_index_W2] - W2_av_mjd_date[before_SDSS_index_W2]))*W2_av_uncs_flux[after_SDSS_index_W2])**2)
            
            if ninety_first_DESI_W2 == 1:
                W2_DESI_interp = W2_averages_flux[0]
                W2_DESI_unc_interp = W2_av_uncs_flux[0]
            elif ninety_last_DESI_W2 == 1:
                W2_DESI_interp = W2_averages_flux[-1]
                W2_DESI_unc_interp = W2_av_uncs_flux[-1]
            elif ninety_before_DESI_W2 == 1:
                W2_DESI_interp = W2_averages_flux[after_DESI_index_W2]
                W2_DESI_unc_interp = W2_av_uncs_flux[after_DESI_index_W2]
            elif ninety_after_DESI_W2 == 1:
                W2_DESI_interp = W2_averages_flux[before_DESI_index_W2]
                W2_DESI_unc_interp = W2_av_uncs_flux[before_DESI_index_W2]
            else:
                #Linearly interpolating to get interpolated flux on a value in between the data points adjacent to DESI.
                W2_DESI_interp = np.interp(DESI_mjd, W2_av_mjd_date, W2_averages_flux)
                W2_DESI_unc_interp = np.sqrt((((W2_av_mjd_date[after_DESI_index_W2] - DESI_mjd)/(W2_av_mjd_date[after_DESI_index_W2] - W2_av_mjd_date[before_DESI_index_W2]))*W2_av_uncs_flux[before_DESI_index_W2])**2 + (((DESI_mjd - W2_av_mjd_date[before_DESI_index_W2])/(W2_av_mjd_date[after_DESI_index_W2] - W2_av_mjd_date[before_DESI_index_W2]))*W2_av_uncs_flux[after_DESI_index_W2])**2)
            
            W2_abs = abs(W2_SDSS_interp-W2_DESI_interp)
            W2_abs_unc = np.sqrt(W2_SDSS_unc_interp**2 + W2_DESI_unc_interp**2)

            W2_abs_norm = ((W2_abs)/(W2_smallest))
            W2_abs_norm_unc = W2_abs_norm*np.sqrt(((W2_abs_unc)/(W2_abs))**2 + ((W2_smallest_unc)/(W2_smallest))**2)

            W2_z_score_SDSS_DESI = (W2_SDSS_interp-W2_DESI_interp)/(W2_DESI_unc_interp)
            W2_z_score_SDSS_DESI_unc = abs(W2_z_score_SDSS_DESI*((W2_abs_unc)/(W2_abs)))
            W2_z_score_DESI_SDSS = (W2_DESI_interp-W2_SDSS_interp)/(W2_SDSS_unc_interp)
            W2_z_score_DESI_SDSS_unc = abs(W2_z_score_DESI_SDSS*((W2_abs_unc)/(W2_abs)))

            W2_SDSS_DESI.append(W2_z_score_SDSS_DESI)
            W2_SDSS_DESI_unc.append(W2_z_score_SDSS_DESI_unc)
            W2_DESI_SDSS.append(W2_z_score_DESI_SDSS)
            W2_DESI_SDSS_unc.append(W2_z_score_DESI_SDSS_unc)
            W2_abs_change.append(W2_abs)
            W2_abs_change_unc.append(W2_abs_unc)
            W2_abs_change_norm.append(W2_abs_norm)
            W2_abs_change_norm_unc.append(W2_abs_norm_unc)

            W2_first_mjd.append(W2_first)
            W2_last_mjd.append(W2_last)

            zscores = np.sort([abs(W1_z_score_SDSS_DESI), abs(W1_z_score_DESI_SDSS), abs(W2_z_score_SDSS_DESI), abs(W2_z_score_DESI_SDSS)]) #sorts in ascending order, nans at end
            zscore_uncs = np.sort([W1_z_score_SDSS_DESI_unc, W1_z_score_DESI_SDSS_unc, W2_z_score_SDSS_DESI_unc, W2_z_score_DESI_SDSS_unc])
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
            W1_largest_unc = W1_av_uncs_flux[W1_averages_flux.index(W1_largest)] #NOT the largest unc. This is the unc in the largest flux value
            W1_smallest = sorted(W1_averages_flux)[0]
            W1_smallest_unc = W1_av_uncs_flux[W1_averages_flux.index(W1_smallest)]

            W1_high_unc.append(W1_largest_unc)
            W1_low.append(W1_smallest)
            W1_low_unc.append(W1_smallest_unc)

            if ninety_first_SDSS_W1 == 1:
                W1_SDSS_interp = W1_averages_flux[0]
                W1_SDSS_unc_interp = W1_av_uncs_flux[0]
            elif ninety_last_SDSS_W1 == 1:
                W1_SDSS_interp = W1_averages_flux[-1]
                W1_SDSS_unc_interp = W1_av_uncs_flux[-1]
            elif ninety_before_SDSS_W1 == 1:
                W1_SDSS_interp = W1_averages_flux[after_SDSS_index_W1]
                W1_SDSS_unc_interp = W1_av_uncs_flux[after_SDSS_index_W1]
            elif ninety_after_SDSS_W1 == 1:
                W1_SDSS_interp = W1_averages_flux[before_SDSS_index_W1]
                W1_SDSS_unc_interp = W1_av_uncs_flux[before_SDSS_index_W1]
            else:
                #Linearly interpolating to get interpolated flux on a value in between the data points adjacent to SDSS.
                W1_SDSS_interp = np.interp(SDSS_mjd, W1_av_mjd_date, W1_averages_flux)
                W1_SDSS_unc_interp = np.sqrt((((W1_av_mjd_date[after_SDSS_index_W1] - SDSS_mjd)/(W1_av_mjd_date[after_SDSS_index_W1] - W1_av_mjd_date[before_SDSS_index_W1]))*W1_av_uncs_flux[before_SDSS_index_W1])**2 + (((SDSS_mjd - W1_av_mjd_date[before_SDSS_index_W1])/(W1_av_mjd_date[after_SDSS_index_W1] - W1_av_mjd_date[before_SDSS_index_W1]))*W1_av_uncs_flux[after_SDSS_index_W1])**2)
            
            if ninety_first_DESI_W1 == 1:
                W1_DESI_interp = W1_averages_flux[0]
                W1_DESI_unc_interp = W1_av_uncs_flux[0]
            elif ninety_last_DESI_W1 == 1:
                W1_DESI_interp = W1_averages_flux[-1]
                W1_DESI_unc_interp = W1_av_uncs_flux[-1]
            elif ninety_before_DESI_W1 == 1:
                W1_DESI_interp = W1_averages_flux[after_DESI_index_W1]
                W1_DESI_unc_interp = W1_av_uncs_flux[after_DESI_index_W1]
            elif ninety_after_DESI_W1 == 1:
                W1_DESI_interp = W1_averages_flux[before_DESI_index_W1]
                W1_DESI_unc_interp = W1_av_uncs_flux[before_DESI_index_W1]
            else:
                #Linearly interpolating to get interpolated flux on a value in between the data points adjacent to DESI.
                W1_DESI_interp = np.interp(DESI_mjd, W1_av_mjd_date, W1_averages_flux)
                W1_DESI_unc_interp = np.sqrt((((W1_av_mjd_date[after_DESI_index_W1] - DESI_mjd)/(W1_av_mjd_date[after_DESI_index_W1] - W1_av_mjd_date[before_DESI_index_W1]))*W1_av_uncs_flux[before_DESI_index_W1])**2 + (((DESI_mjd - W1_av_mjd_date[before_DESI_index_W1])/(W1_av_mjd_date[after_DESI_index_W1] - W1_av_mjd_date[before_DESI_index_W1]))*W1_av_uncs_flux[after_DESI_index_W1])**2)

            W1_abs = abs(W1_SDSS_interp-W1_DESI_interp)
            print(W1_abs)
            print(W1_SDSS_interp)
            print(W1_DESI_interp)
            W1_abs_unc = np.sqrt(W1_SDSS_unc_interp**2 + W1_DESI_unc_interp**2)

            W1_abs_norm = ((W1_abs)/(W1_smallest))
            W1_abs_norm_unc = W1_abs_norm*np.sqrt(((W1_abs_unc)/(W1_abs))**2 + ((W1_smallest_unc)/(W1_smallest))**2)

            W1_z_score_SDSS_DESI = (W1_SDSS_interp-W1_DESI_interp)/(W1_DESI_unc_interp)
            W1_z_score_SDSS_DESI_unc = abs(W1_z_score_SDSS_DESI*((W1_abs_unc)/(W1_abs)))
            W1_z_score_DESI_SDSS = (W1_DESI_interp-W1_SDSS_interp)/(W1_SDSS_unc_interp)
            W1_z_score_DESI_SDSS_unc = abs(W1_z_score_DESI_SDSS*((W1_abs_unc)/(W1_abs)))

            W1_SDSS_DESI.append(W1_z_score_SDSS_DESI)
            W1_SDSS_DESI_unc.append(W1_z_score_SDSS_DESI_unc)
            W1_DESI_SDSS.append(W1_z_score_DESI_SDSS)
            W1_DESI_SDSS_unc.append(W1_z_score_DESI_SDSS_unc)
            W1_abs_change.append(W1_abs)
            W1_abs_change_unc.append(W1_abs_unc)
            W1_abs_change_norm.append(W1_abs_norm)
            W1_abs_change_norm_unc.append(W1_abs_norm_unc)

            W1_first_mjd.append(W1_first)
            W1_last_mjd.append(W1_last)

            W2_abs_norm = np.nan
            W2_abs_norm_unc = np.nan

            W2_z_score_SDSS_DESI = np.nan
            W2_z_score_SDSS_DESI_unc = np.nan
            W2_z_score_DESI_SDSS = np.nan
            W2_z_score_DESI_SDSS_unc = np.nan

            W2_SDSS_DESI.append(np.nan)
            W2_SDSS_DESI_unc.append(np.nan)
            W2_DESI_SDSS.append(np.nan)
            W2_DESI_SDSS_unc.append(np.nan)
            W2_abs_change.append(np.nan)
            W2_abs_change_unc.append(np.nan)
            W2_abs_change_norm.append(np.nan)
            W2_abs_change_norm_unc.append(np.nan)

            W2_first_mjd.append(np.nan)
            W2_last_mjd.append(np.nan)

            zscores = np.sort([abs(W1_z_score_SDSS_DESI), abs(W1_z_score_DESI_SDSS), abs(W2_z_score_SDSS_DESI), abs(W2_z_score_DESI_SDSS)]) #sorts in ascending order, nans at end
            zscore_uncs = np.sort([W1_z_score_SDSS_DESI_unc, W1_z_score_DESI_SDSS_unc, W2_z_score_SDSS_DESI_unc, W2_z_score_DESI_SDSS_unc])
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
        if w == 0 and r == 0: #Good W2 if true
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

            W1_abs_norm = np.nan
            W1_abs_norm_unc = np.nan

            W1_z_score_SDSS_DESI = np.nan
            W1_z_score_SDSS_DESI_unc = np.nan
            W1_z_score_DESI_SDSS = np.nan
            W1_z_score_DESI_SDSS_unc = np.nan

            W1_SDSS_DESI.append(np.nan)
            W1_SDSS_DESI_unc.append(np.nan)
            W1_DESI_SDSS.append(np.nan)
            W1_DESI_SDSS_unc.append(np.nan)
            W1_abs_change.append(np.nan)
            W1_abs_change_unc.append(np.nan)
            W1_abs_change_norm.append(np.nan)
            W1_abs_change_norm_unc.append(np.nan)

            W1_first_mjd.append(np.nan)
            W1_last_mjd.append(np.nan)

            W2_largest = sorted(W2_averages_flux, reverse=True)[0]
            W2_largest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_largest)]
            W2_smallest = sorted(W2_averages_flux)[0]
            W2_smallest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_smallest)]

            W2_high_unc.append(W2_largest_unc)
            W2_low.append(W2_smallest)
            W2_low_unc.append(W2_smallest_unc)

            if ninety_first_SDSS_W2 == 1:
                W2_SDSS_interp = W2_averages_flux[0]
                W2_SDSS_unc_interp = W2_av_uncs_flux[0]
            elif ninety_last_SDSS_W2 == 1:
                W2_SDSS_interp = W2_averages_flux[-1]
                W2_SDSS_unc_interp = W2_av_uncs_flux[-1]
            elif ninety_before_SDSS_W2 == 1:
                W2_SDSS_interp = W2_averages_flux[after_SDSS_index_W2]
                W2_SDSS_unc_interp = W2_av_uncs_flux[after_SDSS_index_W2]
            elif ninety_after_SDSS_W2 == 1:
                W2_SDSS_interp = W2_averages_flux[before_SDSS_index_W2]
                W2_SDSS_unc_interp = W2_av_uncs_flux[before_SDSS_index_W2]
            else:
                #Linearly interpolating to get interpolated flux on a value in between the data points adjacent to SDSS.
                W2_SDSS_interp = np.interp(SDSS_mjd, W2_av_mjd_date, W2_averages_flux)
                W2_SDSS_unc_interp = np.sqrt((((W2_av_mjd_date[after_SDSS_index_W2] - SDSS_mjd)/(W2_av_mjd_date[after_SDSS_index_W2] - W2_av_mjd_date[before_SDSS_index_W2]))*W2_av_uncs_flux[before_SDSS_index_W2])**2 + (((SDSS_mjd - W2_av_mjd_date[before_SDSS_index_W2])/(W2_av_mjd_date[after_SDSS_index_W2] - W2_av_mjd_date[before_SDSS_index_W2]))*W2_av_uncs_flux[after_SDSS_index_W2])**2)
            
            if ninety_first_DESI_W2 == 1:
                W2_DESI_interp = W2_averages_flux[0]
                W2_DESI_unc_interp = W2_av_uncs_flux[0]
            elif ninety_last_DESI_W2 == 1:
                W2_DESI_interp = W2_averages_flux[-1]
                W2_DESI_unc_interp = W2_av_uncs_flux[-1]
            elif ninety_before_DESI_W2 == 1:
                W2_DESI_interp = W2_averages_flux[after_DESI_index_W2]
                W2_DESI_unc_interp = W2_av_uncs_flux[after_DESI_index_W2]
            elif ninety_after_DESI_W2 == 1:
                W2_DESI_interp = W2_averages_flux[before_DESI_index_W2]
                W2_DESI_unc_interp = W2_av_uncs_flux[before_DESI_index_W2]
            else:
                #Linearly interpolating to get interpolated flux on a value in between the data points adjacent to DESI.
                W2_DESI_interp = np.interp(DESI_mjd, W2_av_mjd_date, W2_averages_flux)
                W2_DESI_unc_interp = np.sqrt((((W2_av_mjd_date[after_DESI_index_W2] - DESI_mjd)/(W2_av_mjd_date[after_DESI_index_W2] - W2_av_mjd_date[before_DESI_index_W2]))*W2_av_uncs_flux[before_DESI_index_W2])**2 + (((DESI_mjd - W2_av_mjd_date[before_DESI_index_W2])/(W2_av_mjd_date[after_DESI_index_W2] - W2_av_mjd_date[before_DESI_index_W2]))*W2_av_uncs_flux[after_DESI_index_W2])**2)

            W2_abs = abs(W2_SDSS_interp-W2_DESI_interp)
            W2_abs_unc = np.sqrt(W2_SDSS_unc_interp**2 + W2_DESI_unc_interp**2)

            W2_abs_norm = ((W2_abs)/(W2_smallest))
            W2_abs_norm_unc = W2_abs_norm*np.sqrt(((W2_abs_unc)/(W2_abs))**2 + ((W2_smallest_unc)/(W2_smallest))**2)

            W2_z_score_SDSS_DESI = (W2_SDSS_interp-W2_DESI_interp)/(W2_DESI_unc_interp)
            W2_z_score_SDSS_DESI_unc = abs(W2_z_score_SDSS_DESI*((W2_abs_unc)/(W2_abs)))
            W2_z_score_DESI_SDSS = (W2_DESI_interp-W2_SDSS_interp)/(W2_SDSS_unc_interp)
            W2_z_score_DESI_SDSS_unc = abs(W2_z_score_DESI_SDSS*((W2_abs_unc)/(W2_abs)))

            W2_SDSS_DESI.append(W2_z_score_SDSS_DESI)
            W2_SDSS_DESI_unc.append(W2_z_score_SDSS_DESI_unc)
            W2_DESI_SDSS.append(W2_z_score_DESI_SDSS)
            W2_DESI_SDSS_unc.append(W2_z_score_DESI_SDSS_unc)
            W2_abs_change.append(W2_abs)
            W2_abs_change_unc.append(W2_abs_unc)
            W2_abs_change_norm.append(W2_abs_norm)
            W2_abs_change_norm_unc.append(W2_abs_norm_unc)

            W2_first_mjd.append(W2_first)
            W2_last_mjd.append(W2_last)

            zscores = np.sort([abs(W2_z_score_SDSS_DESI), abs(W1_z_score_DESI_SDSS), abs(W2_z_score_SDSS_DESI), abs(W2_z_score_DESI_SDSS)]) #sorts in ascending order, nans at end
            zscore_uncs = np.sort([W1_z_score_SDSS_DESI_unc, W1_z_score_DESI_SDSS_unc, W2_z_score_SDSS_DESI_unc, W2_z_score_DESI_SDSS_unc])
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

        "W1 Z Score SDSS vs DESI": W1_SDSS_DESI, #1
        "W1 Z Score SDSS vs DESI Unc": W1_SDSS_DESI_unc, #2
        "W1 Z Score DESI vs SDSS": W1_DESI_SDSS, #3
        "W1 Z Score DESI vs SDSS Unc": W1_DESI_SDSS_unc, #4
        "W1 Flux Change": W1_abs_change, #5
        "W1 Flux Change Unc": W1_abs_change_unc, #6
        "W1 NFD": W1_abs_change_norm, #7
        "W1 NFD Unc": W1_abs_change_norm_unc, #8

        "W2 Z Score SDSS vs DESI": W2_SDSS_DESI, #9
        "W2 Z Score SDSS vs DESI Unc": W2_SDSS_DESI_unc, #10
        "W2 Z Score DESI vs SDSS": W2_DESI_SDSS, #11
        "W2 Z Score DESI vs SDSS Unc": W2_DESI_SDSS_unc, #12
        "W2 Flux Change": W2_abs_change, #13
        "W2 Flux Change Unc": W2_abs_change_unc, #14
        "W2 NFD": W2_abs_change_norm, #15
        "W2 NFD Unc": W2_abs_change_norm_unc, #16

        "Mean Z Score": mean_zscore, #17
        "Mean Z Score Unc": mean_zscore_unc, #18
        "Mean NFD": mean_NFD, #19
        "Mean NFD Unc": mean_NFD_unc, #20
        "Median UV NFD On-Off": median_UV_NFD, #21
        "Median UV NFD On-Off Unc": median_UV_NFD_unc, #22

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
        "W1 Mean Unc Counter": W1_mean_uncs, #39
        "W2 Mean Unc Counter": W2_mean_uncs, #40
    }

    # Convert the data into a DataFrame
    df = pd.DataFrame(quantifying_change_data)

    if my_object == 0:
        df.to_csv(f"AGN_Quantifying_Change_Sample_{my_sample}_UV{optical_analysis}.csv", index=False)
    elif my_object == 1:
        df.to_csv(f"CLAGN_Quantifying_Change_UV{optical_analysis}.csv", index=False)