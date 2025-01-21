import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from astropy import units as u #In Astropy, a Quantity object combines a numerical value (like a 1D array of flux) with a physical unit (like W/m^2, erg/s, etc.)
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa

c = 299792458

AGN_sample = pd.read_csv("AGN_Sample.csv")
Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")

AGN_outlier_flux = pd.read_excel('AGN_outlier_flux.xlsx')
AGN_outlier_flux_names = AGN_outlier_flux.iloc[:, 0].tolist()
AGN_outlier_flux_band = AGN_outlier_flux.iloc[:, 1]
AGN_outlier_flux_epoch = AGN_outlier_flux.iloc[:, 2]
CLAGN_outlier_flux = pd.read_excel('CLAGN_outlier_flux.xlsx')
CLAGN_outlier_flux_names = CLAGN_outlier_flux.iloc[:, 0].tolist()
CLAGN_outlier_flux_band = CLAGN_outlier_flux.iloc[:, 1]
CLAGN_outlier_flux_epoch = CLAGN_outlier_flux.iloc[:, 2]

my_object = 0 #0 = AGN. 1 = CLAGN
if my_object == 0:
    object_names = AGN_sample.iloc[:, 3]
elif my_object == 1:
    object_names = [object_name for object_name in Guo_table4.iloc[:, 0] if pd.notna(object_name)]

save_figures = 1 #set to 1 to save figures

def flux(mag, k, wavel): # k is the zero magnitude flux density. For W1 & W2, taken from a data table on the search website - https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
    k = (k*(10**(-6))*(c*10**(10)))/(wavel**2) # converting from Jansky to 10-17 ergs/s/cm2/Å. Express c in Angstrom units
    return k*10**(-mag/2.5)

W1_k = 309.540 #Janskys. This means that mag 0 = 309.540 Janskys at the W1 wl.
W2_k = 171.787
W1_wl = 3.4e4 #Angstroms
W2_wl = 4.6e4

object_names_list = [] #Keeps track of objects that met MIR data requirements to take z score & absolute change

# z_score & absolute change lists
W1_max = []
W1_max_unc = []
W1_min = []
W1_min_unc = []
W1_low = []
W1_median_dev = []
W1_median_unc = []
W1_median_dev_unc = []
W1_abs_change = []
W1_abs_change_unc = []
W1_abs_change_norm = []
W1_abs_change_norm_unc = []
W1_gap = []
W1_epochs = []

W2_max = []
W2_max_unc = []
W2_min = []
W2_min_unc = []
W2_low = []
W2_median_dev = []
W2_median_unc = []
W2_median_dev_unc = []
W2_abs_change = []
W2_abs_change_unc = []
W2_abs_change_norm = []
W2_abs_change_norm_unc = []
W2_gap = []
W2_epochs = []

mean_zscore = []
mean_zscore_unc = []
mean_norm_flux_change = []
mean_norm_flux_change_unc = []

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
object_names = ['125731.87+272313.3']
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
        object_data = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
        SDSS_RA = object_data.iloc[0, 1]
        SDSS_DEC = object_data.iloc[0, 2]

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

    filtered_NEO_rows = NEO_data[(NEO_data.iloc[:, 37] == 1) & (NEO_data.iloc[:, 38] > 5)] #checking for rows where qi_fact == 1 & separation of the WISE instrument to the South Atlantic Anomaly is > 5 degrees
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
    W1_unc = filtered_WISE_rows.iloc[:, 12].tolist() + filtered_NEO_rows_W1.iloc[:, 19].tolist()
    W1_mag = list(zip(W1_mag, mjd_date_W1, W1_unc))
    W1_mag = [tup for tup in W1_mag if not np.isnan(tup[0])] #removing instances where the mag value is NaN

    mjd_date_W2 = filtered_WISE_rows.iloc[:, 10].tolist() + filtered_NEO_rows_W2.iloc[:, 42].tolist()
    W2_mag = filtered_WISE_rows.iloc[:, 14].tolist() + filtered_NEO_rows_W2.iloc[:, 22].tolist()
    W2_unc = filtered_WISE_rows.iloc[:, 15].tolist() + filtered_NEO_rows_W2.iloc[:, 23].tolist()
    W2_mag = list(zip(W2_mag, mjd_date_W2, W2_unc))
    W2_mag = [tup for tup in W2_mag if not np.isnan(tup[0])]

    if len(W1_mag) < 2 and len(W2_mag) < 2: #checking if there is enough data
        print('No W1 & W2 data')
        continue

    #Below code sorts MIR data.
    #Two assumptions required for code to work:
    #1. The data is sorted in order of oldest mjd to most recent.
    #2. There are 2 or more data points.

    # W1 data first
    if len(W1_mag) > 1:
        W1_list = []
        W1_unc_list = []
        W1_mjds = []
        W1_data = []
        for i in range(len(W1_mag)):
            if i == 0: #first reading - store and move on
                W1_list.append(W1_mag[i][0])
                W1_mjds.append(W1_mag[i][1])
                W1_unc_list.append(W1_mag[i][2])
                continue
            elif i == len(W1_mag) - 1: #final data point
                if W1_mag[i][1] - W1_mag[i-1][1] < 100: #checking if final data point is in the same epoch as previous
                    W1_list.append(W1_mag[i][0])
                    W1_mjds.append(W1_mag[i][1])
                    W1_unc_list.append(W1_mag[i][2])

                    # #Median unc
                    # if len(W1_list) > 1:
                    #     W1_data.append( ( np.median(W1_list), np.median(W1_mjds), median_abs_deviation(W1_list) ) )
                    # else:
                    #     W1_data.append( ( np.median(W1_list), np.median(W1_mjds), W1_unc_list[0] ) )

                    #max unc:
                    mean_unc = (1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list)))
                    median_unc = median_abs_deviation(W1_list)
                    W1_data.append( ( np.median(W1_list), np.median(W1_mjds), max(mean_unc, median_unc) ) )
                    continue
                else: #final data point is in an epoch of its own
                    # #median unc
                    # if len(W1_list) > 1:
                    #     W1_data.append( ( np.median(W1_list), np.median(W1_mjds), median_abs_deviation(W1_list) ) )
                    # else:
                    #     W1_data.append( ( np.median(W1_list), np.median(W1_mjds), W1_unc_list[0] ) )

                    #Max unc
                    mean_unc = (1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list)))
                    median_unc = median_abs_deviation(W1_list)
                    W1_data.append( ( np.median(W1_list), np.median(W1_mjds), max(mean_unc, median_unc) ) )

                    W1_data.append( ( np.median(W1_mag[i][0]), np.median(W1_mag[i][1]), W1_mag[i][2] ) )
                    continue
            elif W1_mag[i][1] - W1_mag[i-1][1] < 100: #checking in the same epoch (<100 days between measurements)
                W1_list.append(W1_mag[i][0])
                W1_mjds.append(W1_mag[i][1])
                W1_unc_list.append(W1_mag[i][2])
                continue
            else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                # #Median Unc
                # if len(W1_list) > 1:
                #     W1_data.append( ( np.median(W1_list), np.median(W1_mjds), median_abs_deviation(W1_list) ) )
                # else:
                #     W1_data.append( ( np.median(W1_list), np.median(W1_mjds), W1_unc_list[0] ) )

                #Max unc
                mean_unc = (1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list)))
                median_unc = median_abs_deviation(W1_list)
                W1_data.append( ( np.median(W1_list), np.median(W1_mjds), max(mean_unc, median_unc) ) )

                W1_list = []
                W1_mjds = []
                W1_unc_list = []
                W1_list.append(W1_mag[i][0])
                W1_mjds.append(W1_mag[i][1])
                W1_unc_list.append(W1_mag[i][2])
                continue
        #out of for loop now
    else:
        W1_data = [ (0,0,0) ]

    # W2 data second
    if len(W2_mag) > 1:
        W2_list = []
        W2_unc_list = []
        W2_mjds = []
        W2_data = []
        for i in range(len(W2_mag)):
            if i == 0: #first reading - store and move on
                W2_list.append(W2_mag[i][0])
                W2_mjds.append(W2_mag[i][1])
                W2_unc_list.append(W2_mag[i][2])
                continue
            elif i == len(W2_mag) - 1: #if final data point, close the epoch
                if W2_mag[i][1] - W2_mag[i-1][1] < 100: #checking if final data point is in the same epoch as previous
                    W2_list.append(W2_mag[i][0])
                    W2_mjds.append(W2_mag[i][1])
                    W2_unc_list.append(W2_mag[i][2])

                    # #Median Unc
                    # if len(W2_list) > 1:
                    #     W2_data.append( ( np.median(W2_list), np.median(W2_mjds), median_abs_deviation(W2_list) ) )
                    # else:
                    #     W2_data.append( ( np.median(W2_list), np.median(W2_mjds), W2_unc_list[0] ) )

                    #max unc:
                    mean_unc = (1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list)))
                    median_unc = median_abs_deviation(W2_list)
                    W2_data.append( ( np.median(W2_list), np.median(W2_mjds), max(mean_unc, median_unc) ) )
                    continue
                else: #final data point is in an epoch of its own
                    # #Median Unc
                    # if len(W2_list) > 1:
                    #     W2_data.append( ( np.median(W2_list), np.median(W2_mjds), median_abs_deviation(W2_list) ) )
                    # else:
                    #     W2_data.append( ( np.median(W2_list), np.median(W2_mjds), W2_unc_list[0] ) )

                    #max Unc
                    mean_unc = (1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list)))
                    median_unc = median_abs_deviation(W2_list)
                    W2_data.append( ( np.median(W2_list), np.median(W2_mjds), max(mean_unc, median_unc) ) )

                    W2_data.append( ( np.median(W2_mag[i][0]), np.median(W2_mag[i][1]), W2_mag[i][2] ) )
                    continue
            elif W2_mag[i][1] - W2_mag[i-1][1] < 100: #checking in the same epoch (<100 days between measurements)
                W2_list.append(W2_mag[i][0])
                W2_mjds.append(W2_mag[i][1])
                W2_unc_list.append(W2_mag[i][2])
                continue
            else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                # #Median Unc
                # if len(W2_list) > 1:
                #     W2_data.append( ( np.median(W2_list), np.median(W2_mjds), median_abs_deviation(W2_list) ) )
                # else:
                #     W2_data.append( ( np.median(W2_list), np.median(W2_mjds), W2_unc_list[0] ) )

                #max Unc
                mean_unc = (1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list)))
                median_unc = median_abs_deviation(W2_list)
                W2_data.append( ( np.median(W2_list), np.median(W2_mjds), max(mean_unc, median_unc) ) )

                W2_list = []
                W2_mjds = []
                W2_unc_list = []
                W2_list.append(W2_mag[i][0])
                W2_mjds.append(W2_mag[i][1])
                W2_unc_list.append(W2_mag[i][2])
                continue
    else:
        W2_data = [ (0,0,0) ]

    #removing some epochs:
    if my_object == 0:
        if object_name in AGN_outlier_flux_names:
            AGN_outlier_indices = [i for i, name in enumerate(AGN_outlier_flux_names) if name == object_name]
            if len(AGN_outlier_indices) == 1:
                #1 bad epoch for this object        
                index = AGN_outlier_indices[0]
                if AGN_outlier_flux_band[index] == 'W1':
                    del W1_data[AGN_outlier_flux_epoch[index]-1] #-1 because when I counted epochs I counted the 1st epoch as 1 not 0.
                elif AGN_outlier_flux_band[index] == 'W2':
                    del W2_data[AGN_outlier_flux_epoch[index]-1]

            elif len(AGN_outlier_indices) == 2:
                #2 bad epochs for this object        
                index_one = AGN_outlier_indices[0]
                index_two = AGN_outlier_indices[1]
                if AGN_outlier_flux_band[index_one] == 'W1':
                    del W1_data[AGN_outlier_flux_epoch[index_one]-1]
                    if AGN_outlier_flux_band[index_two] == 'W1':
                        if AGN_outlier_flux_epoch[index_one] < AGN_outlier_flux_epoch[index_two]:
                            del W1_data[AGN_outlier_flux_epoch[index_two]-2]
                        else:
                            del W1_data[AGN_outlier_flux_epoch[index_two]-1]
                    elif AGN_outlier_flux_band[index_two] == 'W2':
                        del W2_data[AGN_outlier_flux_epoch[index_two]-1]

                elif AGN_outlier_flux_band[index_one] == 'W2':
                    del W2_data[AGN_outlier_flux_epoch[index_one]-1]
                    if AGN_outlier_flux_band[index_two] == 'W2':
                        if AGN_outlier_flux_epoch[index_one] < AGN_outlier_flux_epoch[index_two]:
                            del W2_data[AGN_outlier_flux_epoch[index_two]-2]
                        else:
                            del W2_data[AGN_outlier_flux_epoch[index_two]-1]
                    elif AGN_outlier_flux_band[index_two] == 'W1':
                        del W1_data[AGN_outlier_flux_epoch[index_two]-1]

    elif my_object == 1:
        if object_name in CLAGN_outlier_flux_names:
            CLAGN_outlier_indices = [i for i, name in enumerate(CLAGN_outlier_flux_names) if name == object_name]
            if len(CLAGN_outlier_indices) == 1:
                #1 bad epoch for this object        
                index = CLAGN_outlier_indices[0]
                if CLAGN_outlier_flux_band[index] == 'W1':
                    del W1_data[CLAGN_outlier_flux_epoch[index]-1] #-1 because when I counted epochs I counted the 1st epoch as 1 not 0.
                elif CLAGN_outlier_flux_band[index] == 'W2':
                    del W2_data[CLAGN_outlier_flux_epoch[index]-1]

            elif len(CLAGN_outlier_indices) == 2:
                #2 bad epochs for this object        
                index_one = CLAGN_outlier_indices[0]
                index_two = CLAGN_outlier_indices[1]
                if CLAGN_outlier_flux_band[index_one] == 'W1':
                    del W1_data[CLAGN_outlier_flux_epoch[index_one]-1]
                    if CLAGN_outlier_flux_band[index_two] == 'W1':
                        if CLAGN_outlier_flux_epoch[index_one] < CLAGN_outlier_flux_epoch[index_two]:
                            del W1_data[CLAGN_outlier_flux_epoch[index_two]-2]
                        else:
                            del W1_data[CLAGN_outlier_flux_epoch[index_two]-1]
                    elif CLAGN_outlier_flux_band[index_two] == 'W2':
                        del W2_data[CLAGN_outlier_flux_epoch[index_two]-1]

                elif CLAGN_outlier_flux_band[index_one] == 'W2':
                    del W2_data[CLAGN_outlier_flux_epoch[index_one]-1]
                    if CLAGN_outlier_flux_band[index_two] == 'W2':
                        if CLAGN_outlier_flux_epoch[index_one] < CLAGN_outlier_flux_epoch[index_two]:
                            del W2_data[CLAGN_outlier_flux_epoch[index_two]-2]
                        else:
                            del W2_data[CLAGN_outlier_flux_epoch[index_two]-1]
                    elif CLAGN_outlier_flux_band[index_two] == 'W1':
                        del W1_data[CLAGN_outlier_flux_epoch[index_two]-1]

    #want a minimum of 9 (out of ~24 possible) epochs to conduct analysis on.
    if len(W1_data) > 8:
        m = 0
    else:
        m = 1
    if len(W2_data) > 8:
        n = 0
    else:
        n = 1
    if m == 1 and n == 1:
        print('Not enough epochs in W1 & W2')
        continue

    if save_figures == 1:
        fig = plt.figure(figsize=(12,7))
    if m == 0 and n == 0:
        min_mjd = min([W1_data[0][1], W2_data[0][1]])
        W1_av_mjd_date = [tup[1] - min_mjd for tup in W1_data]
        W2_av_mjd_date = [tup[1] - min_mjd for tup in W2_data]
        W1_averages_flux = [flux(tup[0], W1_k, W1_wl) for tup in W1_data]
        W1_av_uncs_flux = [((tup[2]*np.log(10))/(2.5))*flux for tup, flux in zip(W1_data, W1_averages_flux)] #See document in week 5 folder for conversion.
        W2_averages_flux = [flux(tup[0], W2_k, W2_wl) for tup in W2_data]
        W2_av_uncs_flux = [((tup[2]*np.log(10))/(2.5))*flux for tup, flux in zip(W2_data, W2_averages_flux)]
        if save_figures == 1:
            plt.errorbar(W2_av_mjd_date, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color = 'red', capsize=5, label = u'W2 (4.6\u03bcm)')
            plt.errorbar(W1_av_mjd_date, W1_averages_flux, yerr=W1_av_uncs_flux, fmt='o', color = 'orange', capsize=5, label = u'W1 (3.4\u03bcm)')
    elif n == 0:
        min_mjd = W2_data[0][1]
        W2_av_mjd_date = [tup[1] - min_mjd for tup in W2_data]
        W2_averages_flux = [flux(tup[0], W2_k, W2_wl) for tup in W2_data]
        W2_av_uncs_flux = [((tup[2]*np.log(10))/(2.5))*flux for tup, flux in zip(W2_data, W2_averages_flux)]
        if save_figures == 1:
            plt.errorbar(W2_av_mjd_date, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color = 'red', capsize=5, label = u'W2 (4.6\u03bcm)')
    elif m == 0:
        min_mjd = W1_data[0][1]
        W1_av_mjd_date = [tup[1] - min_mjd for tup in W1_data]
        W1_averages_flux = [flux(tup[0], W1_k, W1_wl) for tup in W1_data]
        W1_av_uncs_flux = [((tup[2]*np.log(10))/(2.5))*flux for tup, flux in zip(W1_data, W1_averages_flux)]
        if save_figures == 1:
            plt.errorbar(W1_av_mjd_date, W1_averages_flux, yerr=W1_av_uncs_flux, fmt='o', color = 'orange', capsize=5, label = u'W1 (3.4\u03bcm)')

    if save_figures == 1:
        plt.xlabel('Days since first observation', fontsize = 26)
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.ylabel('Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$', fontsize = 26)
        plt.title(f'Flux vs Time (WISEA J{object_name})', fontsize = 28)
        plt.legend(loc = 'upper left', fontsize = 25)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        # if my_object == 0:
        #     fig.savefig(f'C:/Users/ciara/Dropbox/University/University Work/Fourth Year/Project/AGN Figures/{object_name} - Flux vs Time.png', dpi=300, bbox_inches='tight')
        # elif my_object == 1:
        #     fig.savefig(f'C:/Users/ciara/Dropbox/University/University Work/Fourth Year/Project/CLAGN Figures/{object_name} - Flux vs Time.png', dpi=300, bbox_inches='tight')

    if m == 0: #Good W1 if true
        if n == 0: #Good W2 if true
            #Good W1 & W2
            object_names_list.append(object_name)
            W1_epochs.append(len(W1_data))
            W2_epochs.append(len(W2_data))
            
            W1_median_dev.append(median_abs_deviation(W1_averages_flux))
            W1_median_unc.append(np.nanmedian(W1_av_uncs_flux))
            W1_median_dev_unc.append(median_abs_deviation(W1_av_uncs_flux))

            W2_median_dev.append(median_abs_deviation(W2_averages_flux))
            W2_median_unc.append(np.nanmedian(W2_av_uncs_flux))
            W2_median_dev_unc.append(median_abs_deviation(W2_av_uncs_flux))

            W1_largest = sorted(W1_averages_flux, reverse=True)[0]
            W1_largest_unc = W1_av_uncs_flux[W1_averages_flux.index(W1_largest)] #NOT the largest unc. This is the unc in the largest flux value
            W1_smallest = sorted(W1_averages_flux)[0]
            W1_smallest_unc = W1_av_uncs_flux[W1_averages_flux.index(W1_smallest)]

            W1_low.append(W1_smallest)

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

            W1_gap.append(abs(W1_av_mjd_date[W1_averages_flux.index(W1_largest)] - W1_av_mjd_date[W1_averages_flux.index(W1_smallest)]))

            W2_largest = sorted(W2_averages_flux, reverse=True)[0]
            W2_largest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_largest)]
            W2_smallest = sorted(W2_averages_flux)[0]
            W2_smallest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_smallest)]

            W2_low.append(W2_smallest)

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

            W2_gap.append(abs(W2_av_mjd_date[W2_averages_flux.index(W2_largest)] - W2_av_mjd_date[W2_averages_flux.index(W2_smallest)]))

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
                mean_norm_flux_change.append(np.nanmean(norm_f_ch)) #will be nan
                mean_norm_flux_change_unc.append(np.nan)
            elif np.isnan(norm_f_ch[1]) == True:
                mean_norm_flux_change.append(np.nanmean(norm_f_ch)) #will be norm_f_ch[0]
                mean_norm_flux_change_unc.append(abs(norm_f_ch_unc[0]))
            else:
                mean_norm_flux_change.append(np.nanmean(norm_f_ch))
                mean_norm_flux_change_unc.append((1/2)*np.sqrt(sum(unc**2 for unc in norm_f_ch_unc)))
            
        else: 
            #good W1, bad W2
            object_names_list.append(object_name)
            W1_epochs.append(len(W1_data))
            W2_epochs.append(len(W2_data))

            W1_median_dev.append(median_abs_deviation(W1_averages_flux))
            W1_median_unc.append(np.nanmedian(W1_av_uncs_flux))
            W1_median_dev_unc.append(median_abs_deviation(W1_av_uncs_flux))

            W2_low.append(np.nan)
            W2_median_dev.append(np.nan)
            W2_median_unc.append(np.nan)
            W2_median_dev_unc.append(np.nan)

            W1_largest = sorted(W1_averages_flux, reverse=True)[0]
            W1_largest_unc = W1_av_uncs_flux[W1_averages_flux.index(W1_largest)]
            W1_smallest = sorted(W1_averages_flux)[0]
            W1_smallest_unc = W1_av_uncs_flux[W1_averages_flux.index(W1_smallest)]

            W1_low.append(W1_smallest)

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

            W1_gap.append(abs(W1_av_mjd_date[W1_averages_flux.index(W1_largest)] - W1_av_mjd_date[W1_averages_flux.index(W1_smallest)]))

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

            W2_gap.append(np.nan)

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
                mean_norm_flux_change.append(np.nanmean(norm_f_ch)) #will be nan
                mean_norm_flux_change_unc.append(np.nan)
            elif np.isnan(norm_f_ch[1]) == True:
                mean_norm_flux_change.append(np.nanmean(norm_f_ch)) #will be norm_f_ch[0]
                mean_norm_flux_change_unc.append(abs(norm_f_ch_unc[0]))
            else:
                mean_norm_flux_change.append(np.nanmean(norm_f_ch))
                mean_norm_flux_change_unc.append((1/2)*np.sqrt(sum(unc**2 for unc in norm_f_ch_unc)))

    else: #Bad W1
        if n == 0: #Good W2 if true
            #Bad W1, good W2
            object_names_list.append(object_name)
            W1_epochs.append(len(W1_data))
            W2_epochs.append(len(W2_data))
            
            W1_low.append(np.nan)

            W1_median_dev.append(np.nan)
            W1_median_unc.append(np.nan)
            W1_median_dev_unc.append(np.nan)

            W2_median_dev.append(median_abs_deviation(W2_averages_flux))
            W2_median_unc.append(np.nanmedian(W2_av_uncs_flux))
            W2_median_dev_unc.append(median_abs_deviation(W2_av_uncs_flux))

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

            W1_gap.append(np.nan)

            W2_largest = sorted(W2_averages_flux, reverse=True)[0]
            W2_largest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_largest)]
            W2_smallest = sorted(W2_averages_flux)[0]
            W2_smallest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_smallest)]

            W2_low.append(W2_smallest)

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

            W2_gap.append(abs(W2_av_mjd_date[W2_averages_flux.index(W2_largest)] - W2_av_mjd_date[W2_averages_flux.index(W2_smallest)]))

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
                mean_norm_flux_change.append(np.nanmean(norm_f_ch)) #will be nan
                mean_norm_flux_change_unc.append(np.nan)
            elif np.isnan(norm_f_ch[1]) == True:
                mean_norm_flux_change.append(np.nanmean(norm_f_ch)) #will be norm_f_ch[0]
                mean_norm_flux_change_unc.append(abs(norm_f_ch_unc[0]))
            else:
                mean_norm_flux_change.append(np.nanmean(norm_f_ch))
                mean_norm_flux_change_unc.append((1/2)*np.sqrt(sum(unc**2 for unc in norm_f_ch_unc)))
  
        else:
            #bad W1, bad W2. Should've already 'continued' above to save time.
            print('Bad W1 & W2 data')
            continue

# #for loop now ended
quantifying_change_data = {
    "Object": object_names_list, #0

    "W1 Z Score using Max Unc": W1_max, #1
    "Uncertainty in W1 Z Score using Max Unc": W1_max_unc, #2
    "W1 Z Score using Min Unc": W1_min, #3
    "Uncertainty in W1 Z Score using Min Unc": W1_min_unc, #4
    "W1 Flux Change": W1_abs_change, #5
    "W1 Flux Change Unc": W1_abs_change_unc, #6
    "W1 Normalised Flux Change": W1_abs_change_norm, #7
    "W1 Normalised Flux Change Unc": W1_abs_change_norm_unc, #8

    "W2 Z Score using Max Unc": W2_max, #9
    "Uncertainty in W2 Z Score using Max Unc": W2_max_unc, #10
    "W2 Z Score using Min Unc": W2_min, #11
    "Uncertainty in W2 Z Score using Min Unc": W2_min_unc, #12
    "W2 Flux Change": W2_abs_change, #13
    "W2 Flux Change Unc": W2_abs_change_unc, #14
    "W2 Normalised Flux Change": W2_abs_change_norm, #15
    "W2 Normalised Flux Change Unc": W2_abs_change_norm_unc, #16

    "Mean Z Score": mean_zscore, #17
    "Mean Z Score Unc": mean_zscore_unc, #18
    "Mean Normalised Flux Change": mean_norm_flux_change, #19
    "Mean Normalised Flux Change Unc": mean_norm_flux_change_unc, #20

    "W1 Gap": W1_gap, #21
    "W2 Gap": W2_gap, #22
    "W1 Epochs": W1_epochs, #23
    "W2 Epochs": W2_epochs, #24
    "W1 2nd lowest Flux": W1_low, #25
    "W2 2nd lowest Flux": W2_low, #26
    "W1 median_abs_dev of Flux": W1_median_dev, #27
    "W1 Median Unc": W1_median_unc, #28
    "W1 median_abs_dev of Uncs": W1_median_dev_unc, #29
    "W2 median_abs_dev of Flux": W2_median_dev, #30
    "W2 Median Unc": W2_median_unc, #31
    "W2 median_abs_dev of Uncs": W2_median_dev_unc, #32
}

# Convert the data into a DataFrame
df = pd.DataFrame(quantifying_change_data)

# #median unc
# if my_object == 0:
#     df.to_csv("AGN_Quantifying_Change_just_MIR_2nd_biggest_smallest.csv", index=False)
# elif my_object == 1:
#     df.to_csv("CLAGN_Quantifying_Change_just_MIR_2nd_biggest_smallest.csv", index=False)

#max unc:
if my_object == 0:
    df.to_csv("AGN_Quantifying_Change_just_MIR_max_uncs.csv", index=False)
elif my_object == 1:
    df.to_csv("CLAGN_Quantifying_Change_just_MIR_max_uncs.csv", index=False)