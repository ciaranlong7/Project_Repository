import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from astropy import units as u #In Astropy, a Quantity object combines a numerical value (like a 1D array of flux) with a physical unit (like W/m^2, erg/s, etc.)
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa

c = 299792458

my_object = 0 #0 = AGN. 1 = CLAGN
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
    object_names = AGN_sample.iloc[:, 3]
elif my_object == 1:
    object_names = [object_name for object_name in Guo_table4.iloc[:, 0] if pd.notna(object_name)]
    # object_names = object_names[41:]

def flux(mag, k, wavel): # k is the zero magnitude flux density. For W1 & W2, taken from a data table on the search website - https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
    k = (k*(10**(-6))*(c*10**(10)))/(wavel**2) # converting from Jansky to 10-17 ergs/s/cm2/Å. Express c in Angstrom units
    return k*10**(-mag/2.5)

W1_k = 309.540 #Janskys. This means that mag 0 = 309.540 Janskys at the W1 wl.
W2_k = 171.787
W1_wl = 3.4e4 #Angstroms
W2_wl = 4.6e4

def remove_outliers(data, threshold=15):
    """
    Parameters:
    - data: list of tuples [(flux, mjd, unc), ...]
    - threshold: Modified deviation threshold for outlier removal (default=15)

    Returns:
    - list of filtered tuples without outliers
    """
    if not data:
        return data  # Return empty list if input is empty

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

mean_zscore = []
mean_zscore_unc = []
mean_NFD = []
mean_NFD_unc = []

mod_dev_list = []

Min_SNR = 3 #Options are 10, 3, or 2. #A (SNR>10), B (3<SNR<10) or C (2<SNR<3)
if Min_SNR == 10: #Select Min_SNR on line above.
    MIR_SNR = 'A'
elif Min_SNR == 3:
    MIR_SNR = 'B'
elif Min_SNR == 2:
    MIR_SNR = 'C'
else:
    print('select a valid min SNR - 10, 3 or 2.')

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
    #For CLAGN:
    elif my_object == 1:
        object_data = parent_sample[parent_sample.iloc[:, 3] == object_name]
        SDSS_RA = object_data.iloc[0, 0]
        SDSS_DEC = object_data.iloc[0, 1]

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
        W1_all, W1_mod_dev = remove_outliers(W1_all)
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
        W1_mod_dev = np.array([])

    # W2 data second
    W2_mean_unc_counter = []    
    if len(W2_all) > 1:
        W2_all, W2_mod_dev = remove_outliers(W2_all)
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
        W2_mod_dev = np.array([])
    
    mod_dev_list += W1_mod_dev.tolist()
    mod_dev_list += W2_mod_dev.tolist()

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

    if save_figures == 1:
        fig = plt.figure(figsize=(12,7))
    if m == 0 and n == 0:
        min_mjd = min([W1_data[0][1], W2_data[0][1]])
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
        W2_av_mjd_date = [tup[1] - min_mjd for tup in W2_data]
        W2_averages_flux = [tup[0] for tup in W2_data]
        W2_av_uncs_flux = [tup[2] for tup in W2_data]
        if save_figures == 1:
            plt.errorbar(W2_av_mjd_date, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color = 'orange', capsize=5, label = u'W2 (4.6\u03bcm)')
    elif m == 0:
        min_mjd = W1_data[0][1]
        W1_av_mjd_date = [tup[1] - min_mjd for tup in W1_data]
        W1_averages_flux = [tup[0] for tup in W1_data]
        W1_av_uncs_flux = [tup[2] for tup in W1_data]
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
            # fig.savefig(f'C:/Users/ciara/Dropbox/University/University Work/Fourth Year/Project/AGN Figures - Sample {my_sample}/{object_name} - Flux vs Time.png', dpi=300, bbox_inches='tight')
            fig.savefig(f'C:/Users/ciara/Dropbox/University/University Work/Fourth Year/Project/AGN Figures - Extra/{object_name} - Flux vs Time.png', dpi=300, bbox_inches='tight')
        elif my_object == 1:
            fig.savefig(f'C:/Users/ciara/Dropbox/University/University Work/Fourth Year/Project/CLAGN Figures/{object_name} - Flux vs Time.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

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

            W2_largest = sorted(W2_averages_flux, reverse=True)[0]
            W2_largest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_largest)]
            W2_smallest = sorted(W2_averages_flux)[0]
            W2_smallest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_smallest)]

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
            W1_smallest = sorted(W1_averages_flux)[0]
            W1_smallest_unc = W1_av_uncs_flux[W1_averages_flux.index(W1_smallest)]

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

            W2_largest = sorted(W2_averages_flux, reverse=True)[0]
            W2_largest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_largest)]
            W2_smallest = sorted(W2_averages_flux)[0]
            W2_smallest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_smallest)]

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
    
    mod_dev_data = {
        "Modified Deviations": mod_dev_list, #0
    }

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

        #Brackets () indicate the index of the same column in the csv file created with SDSS/DESI UV analysis
        "W1 First mjd": W1_first_mjd, #21 (#25)
        "W1 Last mjd": W1_last_mjd, #22 (#26)
        "W2 First mjd": W2_first_mjd, #23 (#27)
        "W2 Last mjd": W2_last_mjd, #24 (#28)
        "W1 Epochs": W1_epochs, #25 (#29)
        "W2 Epochs": W2_epochs, #26 (#30)
        "W1 Min Flux": W1_low, #27 (#31)
        "W1 Min Flux Unc": W1_low_unc, #28 (#32)
        "W1 Max Flux Unc": W1_high_unc, #29 (#33)
        "W2 Min Flux": W2_low, #30 (#34)
        "W2 Min Flux Unc": W2_low_unc, #31 (#35)
        "W2 Max Flux Unc": W2_high_unc, #32 (#36)
        "W1 median_abs_dev of Flux": W1_median_dev, #33 (#37)
        "W2 median_abs_dev of Flux": W2_median_dev, #34 (#38)
        "W1 Mean Unc Counter": W1_mean_uncs, #35 (#39)
        "W2 Mean Unc Counter": W2_mean_uncs, #36 (#40)
    }

    # Convert the data into a DataFrame
    mod_dev_df = pd.DataFrame(mod_dev_data)
    df = pd.DataFrame(quantifying_change_data)

    #max unc:
    if my_object == 0:
        mod_dev_df.to_csv(f"AGN_mod_dev_Sample_{my_sample}.csv", index=False)
        df.to_csv(f"AGN_Quantifying_Change_just_MIR_max_uncs_Sample_{my_sample}_Extra.csv", index=False)
    elif my_object == 1:
        mod_dev_df.to_csv("CLAGN_mod_dev.csv", index=False)
        df.to_csv("CLAGN_Quantifying_Change_just_MIR_max_uncs_Extra.csv", index=False)