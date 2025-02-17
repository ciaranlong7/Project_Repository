import numpy as np
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt
import pandas as pd

Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")
my_sample = 1 #set which AGN sample you want
brightness = 2 #0: dim only objects. 1: bright only objects. 2: all objects
my_redshift = 3 #0=low. 1=medium. 2=high. 3=don't filter
MIR_UV = 0 #0=UV. 1=MIR only
turn_on_off = 2 #0=turn-off CLAGN. 1=turn-on CLAGN. #2=don't filter

#plots:
main_MIR = 0 #1 if want main zscore and NFD plot.
main_MIR_NFD_hist = 0 #histogram of distribution of NFD for AGN and non-CL AGN
main_MIR_NFD_hist_bright_dim = 0 #histogram of distribution of NFD for both bright and dim AGN and non-CL AGN
main_MIR_Zs_hist = 0 #histogram of distribution of Z-score for AGN and non-CL AGN
main_MIR_Zs_hist_bright_dim = 0 #histogram of distribution of NFD for both bright and dim AGN and non-CL AGN
UV_MIRZ = 0 #plot of UV normalised flux difference vs z score
UV_MIR_NFD = 0 #plot of UV NFD vs MIR NFD
UVZ_MIRZ = 0 #plot of UV z-score vs MIR z-score
UVNFD_MIRNFD = 0 #plot of UV NFD vs MIR NFD
zs_W1_low = 0 #plot of zscore vs W1 low flux
zs_W2_low = 0 #plot of zscore vs W2 low flux
NFD_W1_low = 0 #plot of NFD vs W1 low flux
NFD_W2_low = 0 #plot of NFD vs W2 low flux
W1_vs_W2_NFD = 0 #plot of W1 NFD vs W2 NFD
W1_vs_W2_Zs = 0 #plot of W1 Zs vs W2 Zs
Modified_Dev_plot = 1 #plot of distribution of modified deviations
Log_Modified_Dev_plot = 0 #same plot as Modified_Dev_plot but with a log scale
epochs_NFD_W1 = 0 #W1 NFD vs W1 epochs
epochs_NFD_W2 = 0 #W2 NFD vs W2 epochs
epochs_zs_W1 = 0 #W1 Zs vs W1 epochs
epochs_zs_W2 = 0 #W2 Zs vs W2 epochs
redshift_dist = 0 #hist of redshift distribution for objects analysed

parent_sample = pd.read_csv('guo23_parent_sample_no_duplicates.csv')
Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")
if my_sample == 1:
    AGN_sample = pd.read_csv("AGN_Sample.csv")
if my_sample == 2:
    AGN_sample = pd.read_csv("AGN_Sample_two.csv")
if my_sample == 3:
    AGN_sample = pd.read_csv("AGN_Sample_three.csv")

CLAGN_names = [object_name for object_name in Guo_table4.iloc[:, 0] if pd.notna(object_name)]
AGN_names = AGN_sample.iloc[:, 3]

CLAGN_redshifts = []
for object_name in CLAGN_names:
    object_row = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
    redshift = object_row.iloc[0, 3]
    CLAGN_redshifts.append(redshift)

AGN_redshifts = []
for object_name in AGN_names:
    object_data = AGN_sample[AGN_sample.iloc[:, 3] == object_name]
    redshift = object_data.iloc[0, 2]
    AGN_redshifts.append(redshift)

median_CLAGN_redshift = np.nanmedian(CLAGN_redshifts)
median_AGN_redshift = np.nanmedian(AGN_redshifts)
print(f'Median CLAGN sample redshift = {median_CLAGN_redshift:.3f}')
print(f'Median AGN sample redshift = {median_AGN_redshift:.3f}')

#Quantifying change data - With UV
if MIR_UV == 0:
    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_UV1.csv')
    #filter for only turn-on or turn-off CLAGN:
    if turn_on_off == 0:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4[Guo_table4['transition'] == 'turn-on'].iloc[:, 0])]
    elif turn_on_off == 1:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4[Guo_table4['transition'] == 'turn-off'].iloc[:, 0])]
    #Drop columns where no UV analysis was performed:
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data.dropna(subset=[CLAGN_quantifying_change_data.columns[21]])
    if brightness == 1:
        # Only objects >= 0.5 min flux in W1 band
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 31] >= 0.5]
    elif brightness == 0:
        # Only objects < 0.5 min flux in W1 band
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 31] < 0.5]
    #Redshift splitting:
    if my_redshift == 0:
        # Low redshift:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 41] <= 0.9]
    elif my_redshift == 1:
        # Medium redshift:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[(CLAGN_quantifying_change_data.iloc[:, 41] > 0.9) & (CLAGN_quantifying_change_data.iloc[:, 41] <= 1.8)]
    elif my_redshift == 2:
        # High Redshift
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 41] > 1.8]

    CLAGN_norm_flux_diff_UV = CLAGN_quantifying_change_data.iloc[:, 21].tolist()
    CLAGN_norm_flux_diff_UV_unc = CLAGN_quantifying_change_data.iloc[:, 22].tolist()
    CLAGN_W1_low_flux = CLAGN_quantifying_change_data.iloc[:, 31].tolist()
    CLAGN_W1_low_flux_unc = CLAGN_quantifying_change_data.iloc[:, 32].tolist()
    CLAGN_W1_high_flux_unc = CLAGN_quantifying_change_data.iloc[:, 33].tolist()
    CLAGN_W1_median_dev_flux = CLAGN_quantifying_change_data.iloc[:, 37].tolist() #median_abs_dev of flux for an object in W1 band
    CLAGN_W2_low_flux = CLAGN_quantifying_change_data.iloc[:, 34].tolist()
    CLAGN_W2_low_flux_unc = CLAGN_quantifying_change_data.iloc[:, 35].tolist()
    CLAGN_W2_high_flux_unc = CLAGN_quantifying_change_data.iloc[:, 36].tolist()
    CLAGN_W2_median_dev_flux = CLAGN_quantifying_change_data.iloc[:, 38].tolist()
    CLAGN_W1_epochs = CLAGN_quantifying_change_data.iloc[:, 29].tolist()
    CLAGN_W2_epochs = CLAGN_quantifying_change_data.iloc[:, 30].tolist()

#Quantifying change data - Just MIR
elif MIR_UV == 1:
    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    if turn_on_off == 0:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4[Guo_table4['transition'] == 'turn-on'].iloc[:, 0])]
    elif turn_on_off == 1:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4[Guo_table4['transition'] == 'turn-off'].iloc[:, 0])]
    if brightness == 1:
        # Only objects >= 0.5 min flux in W1 band
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 27] >= 0.5]
    elif brightness == 0:
        # Only objects < 0.5 min flux in W1 band
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 27] < 0.5]
    #Redshift splitting:
    if my_redshift == 0:
        # Low redshift:
        # CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 37] <= 0.9]
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 35] <= 0.9]
    elif my_redshift == 1:
        # Medium redshift:
        # CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[(CLAGN_quantifying_change_data.iloc[:, 37] > 0.9) & (CLAGN_quantifying_change_data.iloc[:, 37] <= 1.8)]
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[(CLAGN_quantifying_change_data.iloc[:, 35] > 0.9) & (CLAGN_quantifying_change_data.iloc[:, 35] <= 1.8)]
    elif my_redshift == 2:
        # High Redshift
        # CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 37] > 1.8]
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 35] > 1.8]

    CLAGN_W1_low_flux = CLAGN_quantifying_change_data.iloc[:, 27].tolist()
    CLAGN_W1_low_flux_unc = CLAGN_quantifying_change_data.iloc[:, 28].tolist()
    CLAGN_W1_high_flux_unc = CLAGN_quantifying_change_data.iloc[:, 29].tolist()
    CLAGN_W1_median_dev_flux = CLAGN_quantifying_change_data.iloc[:, 33].tolist() #median_abs_dev of flux for an object in W1 band
    CLAGN_W2_low_flux = CLAGN_quantifying_change_data.iloc[:, 30].tolist()
    CLAGN_W2_low_flux_unc = CLAGN_quantifying_change_data.iloc[:, 31].tolist()
    CLAGN_W2_high_flux_unc = CLAGN_quantifying_change_data.iloc[:, 32].tolist()
    CLAGN_W2_median_dev_flux = CLAGN_quantifying_change_data.iloc[:, 34].tolist()
    CLAGN_W1_epochs = CLAGN_quantifying_change_data.iloc[:, 25].tolist()
    CLAGN_W2_epochs = CLAGN_quantifying_change_data.iloc[:, 26].tolist()

#Quantifying change data - Both With UV and Just MIR
print(f'Number of CLAGN Analysed: {len(CLAGN_quantifying_change_data)}')
CLAGN_zscores = CLAGN_quantifying_change_data.iloc[:, 17].tolist()  # 18th column
CLAGN_zscore_uncs = CLAGN_quantifying_change_data.iloc[:, 18].tolist()
CLAGN_norm_flux_diff = CLAGN_quantifying_change_data.iloc[:, 19].tolist()
CLAGN_norm_flux_diff_unc = CLAGN_quantifying_change_data.iloc[:, 20].tolist()
CLAGN_W1_zscore_max = CLAGN_quantifying_change_data.iloc[:, 1].tolist()
CLAGN_W1_zscore_min = CLAGN_quantifying_change_data.iloc[:, 3].tolist()
CLAGN_W1_zscore_mean = [
    np.nanmean([abs(zmax), abs(zmin)]) 
    if not (np.isnan(zmax) and np.isnan(zmin))  # Check if both are NaN
    else np.nan  # Assign NaN if both are NaN
    for zmax, zmin in zip(CLAGN_W1_zscore_max, CLAGN_W1_zscore_min)
]
CLAGN_W1_NFD = CLAGN_quantifying_change_data.iloc[:, 7].tolist()
CLAGN_W2_zscore_max = CLAGN_quantifying_change_data.iloc[:, 9].tolist()
CLAGN_W2_zscore_min = CLAGN_quantifying_change_data.iloc[:, 11].tolist()
CLAGN_W2_zscore_mean = [
    np.nanmean([abs(zmax), abs(zmin)]) 
    if not (np.isnan(zmax) and np.isnan(zmin))  # Check if both are NaN
    else np.nan  # Assign NaN if both are NaN
    for zmax, zmin in zip(CLAGN_W2_zscore_max, CLAGN_W2_zscore_min)
]
CLAGN_W2_NFD = CLAGN_quantifying_change_data.iloc[:, 15].tolist()

CLAGN_names_analysis = CLAGN_quantifying_change_data.iloc[:, 0].tolist()

CLAGN_mod_dev = pd.read_csv('CLAGN_mod_dev.csv')
CLAGN_mod_dev_list = CLAGN_mod_dev.iloc[:, 0].tolist()


#Quantifying change data - With UV
if MIR_UV == 0:
    AGN_quantifying_change_data = pd.read_csv(f'AGN_Quantifying_Change_Sample_{my_sample}_UV1.csv')
    #Drop columns where no UV analysis was performed:
    AGN_quantifying_change_data = AGN_quantifying_change_data.dropna(subset=[AGN_quantifying_change_data.columns[21]])
    if brightness == 1:
        AGN_quantifying_change_data = AGN_quantifying_change_data[AGN_quantifying_change_data.iloc[:, 31] >= 0.5]
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 31] >= 0.5]
    elif brightness == 0:
        AGN_quantifying_change_data = AGN_quantifying_change_data[AGN_quantifying_change_data.iloc[:, 31] < 0.5]
    #Redshift splitting:
    if my_redshift == 0:
        AGN_quantifying_change_data = AGN_quantifying_change_data[AGN_quantifying_change_data.iloc[:, 41] <= 0.9]
    elif my_redshift == 1:
        AGN_quantifying_change_data = AGN_quantifying_change_data[(AGN_quantifying_change_data.iloc[:, 41] > 0.9) & (AGN_quantifying_change_data.iloc[:, 41] > 1.8)]
    elif my_redshift == 2:
        AGN_quantifying_change_data = AGN_quantifying_change_data[AGN_quantifying_change_data.iloc[:, 41] > 1.8]

    AGN_norm_flux_diff_UV = AGN_quantifying_change_data.iloc[:, 21].tolist()
    AGN_norm_flux_diff_UV_unc = AGN_quantifying_change_data.iloc[:, 22].tolist()
    AGN_W1_low_flux = AGN_quantifying_change_data.iloc[:, 31].tolist()
    AGN_W1_low_flux_unc = AGN_quantifying_change_data.iloc[:, 32].tolist()
    AGN_W1_high_flux_unc = AGN_quantifying_change_data.iloc[:, 33].tolist()
    AGN_W1_median_dev_flux = AGN_quantifying_change_data.iloc[:, 37].tolist() #median_abs_dev of flux for an object in W1 band
    AGN_W2_low_flux = AGN_quantifying_change_data.iloc[:, 34].tolist()
    AGN_W2_low_flux_unc = AGN_quantifying_change_data.iloc[:, 35].tolist()
    AGN_W2_high_flux_unc = AGN_quantifying_change_data.iloc[:, 36].tolist()
    AGN_W2_median_dev_flux = AGN_quantifying_change_data.iloc[:, 38].tolist()
    AGN_W1_epochs = AGN_quantifying_change_data.iloc[:, 29].tolist()
    AGN_W2_epochs = AGN_quantifying_change_data.iloc[:, 30].tolist()

#Quantifying change data - Just MIR
elif MIR_UV == 1:
    AGN_quantifying_change_data = pd.read_csv(f'AGN_Quantifying_Change_just_MIR_max_uncs_Sample_{my_sample}.csv')
    if brightness == 1:
        AGN_quantifying_change_data = AGN_quantifying_change_data[AGN_quantifying_change_data.iloc[:, 27] >= 0.5]
    elif brightness == 0:
        AGN_quantifying_change_data = AGN_quantifying_change_data[AGN_quantifying_change_data.iloc[:, 27] < 0.5]
    if my_redshift == 0:
        AGN_quantifying_change_data = AGN_quantifying_change_data[AGN_quantifying_change_data.iloc[:, 37] <= 0.9]
    elif my_redshift == 1:
        AGN_quantifying_change_data = AGN_quantifying_change_data[(AGN_quantifying_change_data.iloc[:, 37] > 0.9) & (AGN_quantifying_change_data.iloc[:, 37] > 1.8)]
    elif my_redshift == 2:
        AGN_quantifying_change_data = AGN_quantifying_change_data[AGN_quantifying_change_data.iloc[:, 37] > 1.8]

    AGN_W1_low_flux = AGN_quantifying_change_data.iloc[:, 27].tolist()
    AGN_W1_low_flux_unc = AGN_quantifying_change_data.iloc[:, 28].tolist()
    AGN_W1_high_flux_unc = AGN_quantifying_change_data.iloc[:, 29].tolist()
    AGN_W1_median_dev_flux = AGN_quantifying_change_data.iloc[:, 33].tolist() #median_abs_dev of flux for an object in W1 band
    AGN_W2_low_flux = AGN_quantifying_change_data.iloc[:, 30].tolist()
    AGN_W2_low_flux_unc = AGN_quantifying_change_data.iloc[:, 31].tolist()
    AGN_W2_high_flux_unc = AGN_quantifying_change_data.iloc[:, 32].tolist()
    AGN_W2_median_dev_flux = AGN_quantifying_change_data.iloc[:, 34].tolist()
    AGN_W1_epochs = AGN_quantifying_change_data.iloc[:, 25].tolist()
    AGN_W2_epochs = AGN_quantifying_change_data.iloc[:, 26].tolist()

#Quantifying change data - Both With UV and Just MIR
print(f'Number of AGN Analysed: {len(AGN_quantifying_change_data)}')
AGN_zscores = AGN_quantifying_change_data.iloc[:, 17].tolist()  # 18th column
AGN_zscore_uncs = AGN_quantifying_change_data.iloc[:, 18].tolist()
AGN_norm_flux_diff = AGN_quantifying_change_data.iloc[:, 19].tolist()
AGN_norm_flux_diff_unc = AGN_quantifying_change_data.iloc[:, 20].tolist()
AGN_W1_zscore_max = AGN_quantifying_change_data.iloc[:, 1].tolist()
AGN_W1_zscore_min = AGN_quantifying_change_data.iloc[:, 3].tolist()
AGN_W1_zscore_mean = [
    np.nanmean([abs(zmax), abs(zmin)]) 
    if not (np.isnan(zmax) and np.isnan(zmin))  # Check if both are NaN
    else np.nan  # Assign NaN if both are NaN
    for zmax, zmin in zip(AGN_W1_zscore_max, AGN_W1_zscore_min)
]
AGN_W1_NFD = AGN_quantifying_change_data.iloc[:, 7].tolist()
AGN_W2_zscore_max = AGN_quantifying_change_data.iloc[:, 9].tolist()
AGN_W2_zscore_min = AGN_quantifying_change_data.iloc[:, 11].tolist()
AGN_W2_zscore_mean = [
    np.nanmean([abs(zmax), abs(zmin)]) 
    if not (np.isnan(zmax) and np.isnan(zmin))  # Check if both are NaN
    else np.nan  # Assign NaN if both are NaN
    for zmax, zmin in zip(AGN_W2_zscore_max, AGN_W2_zscore_min)
]
AGN_W2_NFD = AGN_quantifying_change_data.iloc[:, 15].tolist()

AGN_names_analysis = AGN_quantifying_change_data.iloc[:, 0].tolist()

AGN_mod_dev = pd.read_csv('AGN_mod_dev_Sample_1.csv')
AGN_mod_dev_list = AGN_mod_dev.iloc[:, 0].tolist()

# indices = [i for i, num in enumerate(AGN_zscores) if num > 8 and num < 15]
# for index in indices:
#     print(AGN_names_analysis[index])


print(f'CLAGN W1 median lowest flux = {np.nanmedian(CLAGN_W1_low_flux):.4f}')
print(f'CLAGN W1 median lowest flux Unc = {np.nanmedian(CLAGN_W1_low_flux_unc):.4f}')
print(f'CLAGN W1 median median_abs_dev flux = {np.nanmedian(CLAGN_W1_median_dev_flux):.5f}')
print(f'CLAGN W2 median lowest flux = {np.nanmedian(CLAGN_W2_low_flux):.4f}')
print(f'CLAGN W2 median lowest flux Unc = {np.nanmedian(CLAGN_W2_low_flux_unc):.4f}')
print(f'CLAGN W2 median median_abs_dev flux = {np.nanmedian(CLAGN_W2_median_dev_flux):.5f}')

print(f'AGN W1 median lowest flux = {np.nanmedian(AGN_W1_low_flux):.4f}')
print(f'AGN W1 median lowest flux Unc = {np.nanmedian(AGN_W1_low_flux_unc):.4f}')
print(f'AGN W1 median median_abs_dev flux = {np.nanmedian(AGN_W1_median_dev_flux):.5f}')
print(f'AGN W2 median lowest flux = {np.nanmedian(AGN_W2_low_flux):.4f}')
print(f'AGN W2 median lowest flux Unc = {np.nanmedian(AGN_W2_low_flux_unc):.4f}')
print(f'AGN W2 median median_abs_dev flux = {np.nanmedian(AGN_W2_median_dev_flux):.5f}')


CLAGN_redshifts = []
for object_name in CLAGN_names_analysis:
    object_row = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
    redshift = object_row.iloc[0, 3]
    CLAGN_redshifts.append(redshift)

clean_parent_sample = pd.read_csv('clean_parent_sample_no_CLAGN.csv')
AGN_redshifts = []
for object_name in AGN_names_analysis:
    object_data = clean_parent_sample[clean_parent_sample.iloc[:, 3] == object_name]
    redshift = object_data.iloc[0, 2]
    AGN_redshifts.append(redshift)

median_CLAGN_redshift = np.nanmedian(CLAGN_redshifts)
median_AGN_redshift = np.nanmedian(AGN_redshifts)
print(f'Median CLAGN analysed redshift = {median_CLAGN_redshift:.3f}')
print(f'Median AGN analysed redshift = {median_AGN_redshift:.3f}')

#want the median value of the random sample of AGN
median_norm_flux_diff = np.nanmedian(AGN_norm_flux_diff)
median_norm_flux_diff_unc = np.nanmedian(AGN_norm_flux_diff_unc)
three_sigma_norm_flux_diff = median_norm_flux_diff + 3*median_norm_flux_diff_unc
median_zscore = np.nanmedian(AGN_zscores)
median_zscore_unc = np.nanmedian(AGN_zscore_uncs)
three_sigma_zscore = median_zscore + 3*median_zscore_unc
print(f'Median norm flux difference = {median_norm_flux_diff:.4f}')
print(f'Median norm flux difference unc = {median_norm_flux_diff_unc:.4f}')
print(f'3\u03C3 significance for norm flux difference = {three_sigma_norm_flux_diff:.4f}')
print(f'Median z score = {median_zscore:.4f}')
print(f'Median z score unc = {median_zscore_unc:.4f}')
print(f'3\u03C3 significance for z score = {three_sigma_zscore:.4f}')

median_norm_flux_diff_CLAGN = np.nanmedian(CLAGN_norm_flux_diff)
median_zscore_CLAGN = np.nanmedian(CLAGN_zscores)

i = 0
for zscore in CLAGN_zscores:
    if zscore > three_sigma_zscore:
        i += 1

j = 0
for zscore in AGN_zscores:
    if zscore > three_sigma_zscore:
        j += 1

k = 0
for normdiff in CLAGN_norm_flux_diff:
    if normdiff > three_sigma_norm_flux_diff:
        k += 1

l = 0
for normdiff in AGN_norm_flux_diff:
    if normdiff > three_sigma_norm_flux_diff:
        l += 1

print(f'{k}/{len(CLAGN_norm_flux_diff)}={k/len(CLAGN_norm_flux_diff)*100:.2f}% of CLAGN above NFD threshold')
print(f'{i}/{len(CLAGN_zscores)}={i/len(CLAGN_zscores)*100:.2f}% of CLAGN above zscore threshold')
print(f'{l}/{len(AGN_norm_flux_diff)}={l/len(AGN_norm_flux_diff)*100:.2f}% of AGN above NFD threshold')
print(f'{j}/{len(AGN_zscores)}={j/len(AGN_zscores)*100:.2f}% of AGN above zscore threshold')

if MIR_UV == 0:
    median_UV_NFD = np.nanmedian(AGN_norm_flux_diff_UV)
    median_UV_NFD_unc = np.nanmedian(AGN_norm_flux_diff_UV_unc)
    three_sigma_UV_NFD = median_UV_NFD + 3*median_UV_NFD_unc
    print(f'Median UV NFD = {median_UV_NFD:.4f}')
    print(f'Median UV NFD unc = {median_UV_NFD_unc:.4f}')
    print(f'3\u03C3 significance for UV NFD = {three_sigma_UV_NFD:.4f}')

    b = 0
    for UV_normdiff in CLAGN_norm_flux_diff_UV:
        if UV_normdiff > three_sigma_UV_NFD:
            b += 1
    
    d = 0
    for UV_normdiff in AGN_norm_flux_diff_UV:
        if UV_normdiff > three_sigma_UV_NFD:
            d += 1

    print(f'{b}/{len(CLAGN_norm_flux_diff_UV)}={b/len(CLAGN_norm_flux_diff_UV)*100:.2f}% of CLAGN above UV NFD threshold')
    print(f'{d}/{len(AGN_norm_flux_diff_UV)}={d/len(AGN_norm_flux_diff_UV)*100:.2f}% of AGN above UV NFD threshold')


# ### BELOW INVESTIGATION CHECKS WHETHER ELIMINATING OBJECTS WITH A HIGH UNC RATIO (eg NFD_UNC/NFD) IMPROVES RESULTS.
# ### I find that eliminating objects with a high unc ratio increases the amount of CLAGN & non-CL AGN that are above the thresholds.
# ### However, this is in a proportionate manner. 
# ### I.e., ratio of CLAGN to non CL AGN above the thresholds remains ~10x for NFD and ~3x for zscore.

# CLAGN_norm_uncpc = []
# CLAGN_z_uncpc = []
# test_CLAGN_zscores = []
# test_CLAGN_nfd = []
# for i in range(len(CLAGN_quantifying_change_data)):
#     CLAGN_norm_uncpc.append(CLAGN_norm_flux_diff_unc[i]/CLAGN_norm_flux_diff[i])
#     CLAGN_z_uncpc.append(CLAGN_zscore_uncs[i]/CLAGN_zscores[i])
#     if CLAGN_zscore_uncs[i]/CLAGN_zscores[i] > 0.5:
#         # print(f'CLAGN Unc % = {CLAGN_norm_flux_diff_unc[i]/CLAGN_norm_flux_diff[i]*100:.3f}')
#         # print(f'CLAGN Z Score = {CLAGN_zscores[i]}')
#         # print(f'CLAGN NFD = {CLAGN_norm_flux_diff[i]}')
#         continue
#     else: #unc % <50
#         test_CLAGN_zscores.append(CLAGN_zscores[i])
#         test_CLAGN_nfd.append(CLAGN_norm_flux_diff[i])

# CLAGN_norm_fifty = len([x for x in CLAGN_norm_uncpc if x > 0.5])
# CLAGN_z_fifty = len([x for x in CLAGN_z_uncpc if x > 0.5])

# print(f'{CLAGN_norm_fifty/len(CLAGN_quantifying_change_data)*100:.3f}% of CLAGN have a NFD unc > 50% of NFD value')
# print(f'{CLAGN_z_fifty/len(CLAGN_quantifying_change_data)*100:.3f}% of CLAGN have a zscore unc > 50% of zscore value')

# AGN_norm_uncpc = []
# AGN_z_uncpc = []
# test_AGN_zscores = []
# test_AGN_zscore_unc = []
# test_AGN_nfd = []
# test_AGN_nfd_unc = []
# for i in range(len(AGN_quantifying_change_data)):
#     AGN_norm_uncpc.append(AGN_norm_flux_diff_unc[i]/AGN_norm_flux_diff[i])
#     AGN_z_uncpc.append(AGN_zscore_uncs[i]/AGN_zscores[i])
#     if AGN_zscore_uncs[i]/AGN_zscores[i] > 0.5:
#         # print(f'AGN Unc % = {AGN_norm_flux_diff_unc[i]/AGN_norm_flux_diff[i]*100:.3f}')
#         # print(f'AGN Z Score = {AGN_zscores[i]}')
#         # print(f'AGN NFD = {AGN_norm_flux_diff[i]}')
#         continue
#     else: #unc % <50
#         test_AGN_zscores.append(AGN_zscores[i])
#         test_AGN_zscore_unc.append(AGN_zscore_uncs[i])
#         test_AGN_nfd.append(AGN_norm_flux_diff[i])
#         test_AGN_nfd_unc.append(AGN_norm_flux_diff_unc[i])

# AGN_norm_fifty = len([x for x in AGN_norm_uncpc if x > 0.5])
# AGN_z_fifty = len([x for x in CLAGN_z_uncpc if x > 0.5])

# print(f'{AGN_norm_fifty/len(AGN_quantifying_change_data)*100:.3f}% of AGN have a NFD unc > 50% of NFD value')
# print(f'{AGN_z_fifty/len(AGN_quantifying_change_data)*100:.3f}% of AGN have a zscore unc > 50% of zscore value')

# test_median_norm_flux_diff = np.nanmedian(test_AGN_nfd)
# test_median_norm_flux_diff_unc = np.nanmedian(test_AGN_nfd_unc)
# test_three_sigma_norm_flux_diff = test_median_norm_flux_diff + 3*test_median_norm_flux_diff_unc
# test_median_zscore = np.nanmedian(test_AGN_zscores)
# test_median_zscore_unc = np.nanmedian(test_AGN_zscore_unc)
# test_three_sigma_zscore = test_median_zscore + 3*test_median_zscore_unc

# print(f'Test Median norm flux difference = {test_median_norm_flux_diff:.4f}')
# print(f'Test Median norm flux difference unc = {test_median_norm_flux_diff_unc:.4f}')
# print(f'Test 3\u03C3 significance for norm flux difference = {test_three_sigma_norm_flux_diff:.4f}')
# print(f'Test Median z score = {test_median_zscore:.4f}')
# print(f'Test Median z score unc = {test_median_zscore_unc:.4f}')
# print(f'Test 3\u03C3 significance for z score = {test_three_sigma_zscore:.4f}')

# i = 0
# for zscore in test_CLAGN_zscores:
#     if zscore > test_three_sigma_zscore:
#         i += 1
# j = 0
# for zscore in test_AGN_zscores:
#     if zscore > test_three_sigma_zscore:
#         j += 1
# k = 0
# for normdiff in test_CLAGN_nfd:
#     if normdiff > test_three_sigma_norm_flux_diff:
#         k += 1
# l = 0
# for normdiff in test_AGN_nfd:
#     if normdiff > test_three_sigma_norm_flux_diff:
#         l += 1

# print('Eliminated Objects with unc % > 50:')
# print(f'{i}/{len(test_CLAGN_zscores)}={i/len(test_CLAGN_zscores)*100:.3f}% of CLAGN above zscore threshold')
# print(f'{k}/{len(test_CLAGN_nfd)}={k/len(test_CLAGN_nfd)*100:.3f}% of CLAGN above norm_diff threshold')
# print(f'{j}/{len(test_AGN_zscores)}={j/len(test_AGN_zscores)*100:.3f}% of AGN above zscore threshold')
# print(f'{l}/{len(test_AGN_nfd)}={l/len(test_AGN_nfd)*100:.3f}% of AGN above norm_diff threshold')

# ### END OF INVESTIGATION


# # A histogram of z score values & normalised flux difference values
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))  # Creates a figure with 1 row and 2 columns

# zscore_binsize = 1
# # # #CLAGN
# bins_zscores = np.arange(0, max(CLAGN_zscores)+zscore_binsize, zscore_binsize)
# ax1.hist(CLAGN_zscores, bins=bins_zscores, color='orange', edgecolor='black', label=f'binsize = {zscore_binsize}')
# ax1.axvline(median_zscore_CLAGN, linewidth=2, linestyle='--', color='black', label = f'Median = {median_zscore_CLAGN:.2f}')
# # # # #AGN
# # bins_zscores = np.arange(0, max(AGN_zscores)+zscore_binsize, zscore_binsize)
# # ax1.hist(AGN_zscores, bins=bins_zscores, color='orange', edgecolor='black', label=f'binsize = {zscore_binsize}')
# # ax1.axvline(median_zscore, linewidth=2, linestyle='--', color='black', label = f'Median = {median_zscore:.2f}')
# ax1.set_xlabel('Z Score')
# ax1.set_ylabel('Frequency')
# ax1.legend(loc='upper right')

# norm_flux_diff_binsize = 0.10
# # #CLAGN
# bins_norm_flux_diff = np.arange(0, max(CLAGN_norm_flux_diff)+norm_flux_diff_binsize, norm_flux_diff_binsize)
# ax2.hist(CLAGN_norm_flux_diff, bins=bins_norm_flux_diff, color='blue', edgecolor='black', label=f'binsize = {norm_flux_diff_binsize}')
# ax2.axvline(median_norm_flux_diff_CLAGN, linewidth=2, linestyle='--', color='black', label = f'Median = {median_norm_flux_diff_CLAGN:.2f}')
# # # #AGN
# # bins_norm_flux_diff = np.arange(0, max(AGN_norm_flux_diff)+norm_flux_diff_binsize, norm_flux_diff_binsize)
# # ax2.hist(AGN_norm_flux_diff, bins=bins_norm_flux_diff, color='blue', edgecolor='black', label=f'binsize = {norm_flux_diff_binsize}')
# # ax2.axvline(median_norm_flux_diff, linewidth=2, linestyle='--', color='black', label = f'Median = {median_norm_flux_diff:.2f}')
# ax2.set_xlabel('Normalised Flux Difference')
# ax2.set_ylabel('Frequency')
# ax2.legend(loc='upper right')

# # #CLAGN
# plt.suptitle('Z Score & Normalised Flux Difference Distribution - Guo CLAGN', fontsize=16)
# # # #AGN
# # plt.suptitle('Z Score & Normalised Flux Difference Distribution - Parent Sample AGN', fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the main title
# plt.show()


# # #Creating a 2d plot for normalised flux difference & z score:
if main_MIR == 1:
    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_zscores, AGN_norm_flux_diff, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_zscores, CLAGN_norm_flux_diff, s= 100, color='red',  label='CLAGN')
    # plt.errorbar(AGN_zscores, AGN_norm_flux_diff, xerr=AGN_zscore_uncs, yerr=AGN_norm_flux_diff_unc, fmt='o', color='blue', label='Non-CL AGN')
    # plt.errorbar(CLAGN_zscores, CLAGN_norm_flux_diff, xerr=CLAGN_zscore_uncs, yerr=CLAGN_norm_flux_diff_unc, fmt='o', color='red',  label='CLAGN')
    plt.axhline(y=three_sigma_norm_flux_diff, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2)
    plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
    plt.ylim(0, 1.05*max(CLAGN_norm_flux_diff+AGN_norm_flux_diff))
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xlabel("Z-Score", fontsize = 26)
    plt.ylabel("NFD", fontsize = 26)
    plt.title(f"Characterising MIR Variability in AGN (Sample {my_sample})", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    ax = plt.gca()
    if my_sample == 1:
        plt.text(0.99, 0.46, f'{i/len(CLAGN_zscores)*100:.1f}% CLAGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.4, f'{j/len(AGN_zscores)*100:.1f}% AGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.16, 0.95, f'{k/len(CLAGN_norm_flux_diff)*100:.1f}% CLAGN > NFD Threshold', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.16, 0.89, f'{l/len(AGN_norm_flux_diff)*100:.1f}% AGN > NFD Threshold', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    elif my_sample == 2:
        plt.text(0.99, 0.52, f'{i/len(CLAGN_zscores)*100:.1f}% CLAGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.46, f'{j/len(AGN_zscores)*100:.1f}% AGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.64, f'{k/len(CLAGN_norm_flux_diff)*100:.1f}% CLAGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.58, f'{l/len(AGN_norm_flux_diff)*100:.1f}% AGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    elif my_sample == 3:
        plt.text(0.99, 0.68, f'{i/len(CLAGN_zscores)*100:.1f}% CLAGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.62, f'{j/len(AGN_zscores)*100:.1f}% AGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.81, f'{k/len(CLAGN_norm_flux_diff)*100:.1f}% CLAGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.75, f'{l/len(AGN_norm_flux_diff)*100:.1f}% AGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)

    # The default transform specifies that text is in data coords, alternatively, you can specify text in axis coords 
    # (0,0 is lower-left and 1,1 is upper-right).
    plt.show()


if main_MIR_NFD_hist == 1:
    AGN_flux_diff_binsize = (max(AGN_norm_flux_diff) - min(AGN_norm_flux_diff))/50  # 50 bins
    AGN_bins_flux_diff = np.arange(min(AGN_norm_flux_diff), max(AGN_norm_flux_diff) + AGN_flux_diff_binsize, AGN_flux_diff_binsize)
    MAD_flux_diff = median_abs_deviation(AGN_norm_flux_diff)
    x_start = median_norm_flux_diff - MAD_flux_diff
    x_end = median_norm_flux_diff + MAD_flux_diff
    x_start_threshold = median_norm_flux_diff - 3*median_norm_flux_diff_unc
    x_end_threshold = median_norm_flux_diff + 3*median_norm_flux_diff_unc
    counts, bin_edges = np.histogram(AGN_norm_flux_diff, bins=AGN_bins_flux_diff)
    height = max(counts)/2
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)  
    ax1.hist(AGN_norm_flux_diff, bins=AGN_bins_flux_diff, color='blue', edgecolor='black', label='Non-CL AGN')
    ax1.axvline(median_norm_flux_diff, linewidth=2, linestyle='--', color='darkblue', label=f'Non-CL AGN Median = {median_norm_flux_diff:.2f}')
    ax1.axvline(three_sigma_norm_flux_diff, linewidth=2, linestyle='--', color='black', label=f'{l/len(AGN_norm_flux_diff)*100:.1f}% > Threshold = {three_sigma_norm_flux_diff:.2f}')
    ax1.plot((x_start, x_end), (height, height), linewidth=2, color='black', label = f'Median Absolute Deviation = {MAD_flux_diff:.2f}')
    ax1.plot((x_start_threshold, x_end_threshold), (height+0.5, height+0.5), linewidth=2, color='darkorange', label = f'3X Median Uncertainty = {3*median_norm_flux_diff_unc:.2f}')
    ax1.set_ylabel('Non-CL AGN Frequency', color='black')
    ax1.legend(loc='upper right')

    CLAGN_flux_diff_binsize = (max(CLAGN_norm_flux_diff) - min(CLAGN_norm_flux_diff))/50  # 50 bins
    CLAGN_bins_flux_diff = np.arange(min(CLAGN_norm_flux_diff), max(CLAGN_norm_flux_diff) + CLAGN_flux_diff_binsize, CLAGN_flux_diff_binsize)
    MAD_flux_diff_CLAGN = median_abs_deviation(CLAGN_norm_flux_diff)
    x_start_CLAGN = median_norm_flux_diff_CLAGN - MAD_flux_diff_CLAGN
    x_end_CLAGN = median_norm_flux_diff_CLAGN + MAD_flux_diff_CLAGN
    counts_CLAGN, bin_edges_CLAGN = np.histogram(CLAGN_norm_flux_diff, bins=CLAGN_bins_flux_diff)
    height_CLAGN = max(counts_CLAGN)/2
    
    ax2.hist(CLAGN_norm_flux_diff, bins=CLAGN_bins_flux_diff, color='red', edgecolor='black', label='CLAGN')
    ax2.axvline(median_norm_flux_diff_CLAGN, linewidth=2, linestyle='--', color='darkred', label=f'CLAGN Median = {median_norm_flux_diff_CLAGN:.2f}')
    ax2.axvline(three_sigma_norm_flux_diff, linewidth=2, linestyle='--', color='black', label=f'{k/len(CLAGN_norm_flux_diff)*100:.1f}% > Threshold = {three_sigma_norm_flux_diff:.2f}')
    ax2.plot((x_start_CLAGN, x_end_CLAGN), (height_CLAGN, height_CLAGN), linewidth=2, color='black', label = f'Median Absolute Deviation = {MAD_flux_diff_CLAGN:.2f}')
    ax2.set_xlabel('NFD')
    ax2.set_ylabel('CLAGN Frequency', color='black')
    ax2.legend(loc='upper right')

    if brightness == 0:
        plt.suptitle(f'NFD Distribution (Dim) - CLAGN & Non-CL AGN Sample {my_sample}')
    elif brightness == 1:
        plt.suptitle(f'NFD Distribution (Bright) - CLAGN & Non-CL AGN Sample {my_sample}')
    elif brightness == 2:
        plt.suptitle(f'NFD Distribution (All) - CLAGN & Non-CL AGN Sample {my_sample}')
    plt.tight_layout()
    plt.show()


if main_MIR_NFD_hist_bright_dim == 1:
    AGN_quantifying_change_data = pd.read_csv(f'AGN_Quantifying_Change_just_MIR_max_uncs_Sample_{my_sample}.csv')
    AGN_quantifying_change_data = AGN_quantifying_change_data[AGN_quantifying_change_data.iloc[:, 27] >= 0.5]
    AGN_norm_flux_diff_bright = AGN_quantifying_change_data.iloc[:, 19].tolist()
    AGN_norm_flux_diff_unc_bright = AGN_quantifying_change_data.iloc[:, 20].tolist()
    AGN_quantifying_change_data = pd.read_csv(f'AGN_Quantifying_Change_just_MIR_max_uncs_Sample_{my_sample}.csv')
    AGN_quantifying_change_data = AGN_quantifying_change_data[AGN_quantifying_change_data.iloc[:, 27] < 0.5]
    AGN_norm_flux_diff_dim = AGN_quantifying_change_data.iloc[:, 19].tolist()
    AGN_norm_flux_diff_unc_dim = AGN_quantifying_change_data.iloc[:, 20].tolist()

    AGN_norm_flux_diff_all = AGN_norm_flux_diff_bright+AGN_norm_flux_diff_dim
    median_norm_flux_diff_AGN_bright = np.nanmedian(AGN_norm_flux_diff_bright)
    median_norm_flux_diff_AGN_unc_bright = np.nanmedian(AGN_norm_flux_diff_unc_bright)
    three_sigma_norm_flux_diff_bright = median_norm_flux_diff_AGN_bright + 3*median_norm_flux_diff_AGN_unc_bright
    median_norm_flux_diff_AGN_dim = np.nanmedian(AGN_norm_flux_diff_dim)
    median_norm_flux_diff_AGN_unc_dim = np.nanmedian(AGN_norm_flux_diff_unc_dim)
    three_sigma_norm_flux_diff_dim = median_norm_flux_diff_AGN_dim + 3*median_norm_flux_diff_AGN_unc_dim

    AGN_flux_diff_binsize = (max(AGN_norm_flux_diff_all) - min(AGN_norm_flux_diff_all))/50  # 50 bins
    AGN_bins_flux_diff = np.arange(min(AGN_norm_flux_diff_all), max(AGN_norm_flux_diff_all) + AGN_flux_diff_binsize, AGN_flux_diff_binsize)
    x_start_threshold_bright = median_norm_flux_diff_AGN_bright - 3*median_norm_flux_diff_AGN_unc_bright
    x_end_threshold_bright = median_norm_flux_diff_AGN_bright + 3*median_norm_flux_diff_AGN_unc_bright
    counts_bright, bin_edges = np.histogram(AGN_norm_flux_diff_bright, bins=AGN_bins_flux_diff)
    height_bright = max(counts_bright)/2

    x_start_threshold_dim = median_norm_flux_diff_AGN_dim - 3*median_norm_flux_diff_AGN_unc_dim
    x_end_threshold_dim = median_norm_flux_diff_AGN_dim + 3*median_norm_flux_diff_AGN_unc_dim
    counts_dim, bin_edges = np.histogram(AGN_norm_flux_diff_dim, bins=AGN_bins_flux_diff)
    height_dim = max(counts_dim)/2

    a = 0
    for NFD in AGN_norm_flux_diff_bright:
        if NFD > three_sigma_norm_flux_diff_bright:
            a += 1
    
    b = 0
    for NFD in AGN_norm_flux_diff_dim:
        if NFD > three_sigma_norm_flux_diff_dim:
            b += 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)  
    ax1.hist(AGN_norm_flux_diff_bright, bins=AGN_bins_flux_diff, color='blue', alpha=0.7, edgecolor='black', label='Bright Non-CL AGN')
    ax1.hist(AGN_norm_flux_diff_dim, bins=AGN_bins_flux_diff, color='blueviolet', alpha=0.7, edgecolor='black', label='Dim Non-CL AGN')
    ax1.axvline(median_norm_flux_diff_AGN_bright, linewidth=2, linestyle='--', color='darkblue', label=f'Bright Non-CL AGN Median = {median_norm_flux_diff_AGN_bright:.2f}')
    ax1.axvline(median_norm_flux_diff_AGN_dim, linewidth=2, linestyle='--', color='darkblue', label=f'Dim Non-CL AGN Median = {median_norm_flux_diff_AGN_dim:.2f}')
    ax1.axvline(three_sigma_norm_flux_diff_bright, linewidth=2, linestyle='--', color='black', label=f'{a/len(AGN_norm_flux_diff_bright)*100:.1f}% Bright Non-CL AGN > Bright Threshold = {three_sigma_norm_flux_diff_bright:.2f}')
    ax1.axvline(three_sigma_norm_flux_diff_dim, linewidth=2, linestyle='--', color='grey', label=f'{b/len(AGN_norm_flux_diff_dim)*100:.1f}% Dim Non-CL AGN > Dim Threshold = {three_sigma_norm_flux_diff_dim:.2f}')
    ax1.plot((x_start_threshold_bright, x_end_threshold_bright), (height_bright+0.75, height_bright+0.75), linewidth=2, color='tan', label = f'3X Median Bright Uncertainty = {3*median_norm_flux_diff_AGN_unc_bright:.2f}')
    ax1.plot((x_start_threshold_dim, x_end_threshold_dim), (height_dim+0.25, height_dim+0.25), linewidth=2, color='darkorange', label = f'3X Median Dim Uncertainty = {3*median_norm_flux_diff_AGN_unc_dim:.2f}')
    ax1.set_ylabel('Non-CL AGN Frequency', color='black')
    ax1.legend(loc='upper right')

    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 27] >= 0.5]
    CLAGN_norm_flux_diff_bright = CLAGN_quantifying_change_data.iloc[:, 19].tolist()
    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 27] < 0.5]
    CLAGN_norm_flux_diff_dim = CLAGN_quantifying_change_data.iloc[:, 19].tolist()

    CLAGN_norm_flux_diff_all = CLAGN_norm_flux_diff_bright+CLAGN_norm_flux_diff_dim
    median_norm_flux_diff_CLAGN_bright = np.nanmedian(CLAGN_norm_flux_diff_bright)
    median_norm_flux_diff_CLAGN_dim = np.nanmedian(CLAGN_norm_flux_diff_dim)

    CLAGN_flux_diff_binsize = (max(CLAGN_norm_flux_diff_all) - min(CLAGN_norm_flux_diff_all))/50  # 50 bins
    CLAGN_bins_flux_diff = np.arange(min(CLAGN_norm_flux_diff_all), max(CLAGN_norm_flux_diff_all) + CLAGN_flux_diff_binsize, CLAGN_flux_diff_binsize)
    
    c = 0
    for NFD in CLAGN_norm_flux_diff_bright:
        if NFD > three_sigma_norm_flux_diff_bright:
            c += 1
    
    d = 0
    for NFD in CLAGN_norm_flux_diff_dim:
        if NFD > three_sigma_norm_flux_diff_dim:
            d += 1

    ax2.hist(CLAGN_norm_flux_diff_bright, bins=CLAGN_bins_flux_diff, color='brown', alpha=0.8, edgecolor='black', label='Bright CLAGN')
    ax2.hist(CLAGN_norm_flux_diff_dim, bins=CLAGN_bins_flux_diff, color='salmon', alpha=0.4, edgecolor='black', label='Dim CLAGN')
    ax2.axvline(median_norm_flux_diff_CLAGN_bright, linewidth=2, linestyle='--', color='darkred', label=f'Bright CLAGN Median = {median_norm_flux_diff_CLAGN_bright:.2f}')
    ax2.axvline(median_norm_flux_diff_CLAGN_dim, linewidth=2, linestyle='--', color='darkred', label=f'Dim CLAGN Median = {median_norm_flux_diff_CLAGN_dim:.2f}')
    ax2.axvline(three_sigma_norm_flux_diff_bright, linewidth=2, linestyle='--', color='black', label=f'{c/len(CLAGN_norm_flux_diff_bright)*100:.1f}% Bright CLAGN > Bright Threshold = {three_sigma_norm_flux_diff_bright:.2f}')
    ax2.axvline(three_sigma_norm_flux_diff_dim, linewidth=2, linestyle='--', color='grey', label=f'{d/len(CLAGN_norm_flux_diff_dim)*100:.1f}% Dim CLAGN > Dim Threshold = {three_sigma_norm_flux_diff_dim:.2f}')
    ax2.set_xlabel('NFD')
    ax2.set_ylabel('CLAGN Frequency', color='black')
    ax2.legend(loc='upper right')

    plt.suptitle(f'NFD Distribution - CLAGN & Non-CL AGN Sample {my_sample}')
    plt.tight_layout()
    plt.show()


if main_MIR_Zs_hist == 1:
    AGN_zscore_binsize = (max(AGN_zscores) - min(AGN_zscores))/50  # 50 bins
    AGN_bins_zscore = np.arange(min(AGN_zscores), max(AGN_zscores) + AGN_zscore_binsize, AGN_zscore_binsize)
    MAD_zscore = median_abs_deviation(AGN_zscores)
    x_start = median_zscore - MAD_zscore
    x_end = median_zscore + MAD_zscore
    x_start_threshold = median_zscore - 3*median_zscore_unc
    x_end_threshold = median_zscore + 3*median_zscore_unc
    height = max(counts)/2
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)  
    ax1.hist(AGN_zscores, bins=AGN_bins_zscore, color='blue', edgecolor='black', label='Non-CL AGN')
    ax1.axvline(median_zscore, linewidth=2, linestyle='--', color='darkblue', label=f'Non-CL AGN Median = {median_zscore:.2f}')
    ax1.axvline(three_sigma_zscore, linewidth=2, linestyle='--', color='black', label=f'{j/len(AGN_zscores)*100:.1f}% > Threshold = {three_sigma_zscore:.2f}')
    ax1.plot((x_start, x_end), (height, height), linewidth=2, color='black', label = f'Median Absolute Deviation = {MAD_zscore:.2f}')
    ax1.plot((x_start_threshold, x_end_threshold), (height+0.5, height+0.5), linewidth=2, color='darkorange', label = f'3X Median Uncertainty = {3*median_zscore_unc:.2f}')
    ax1.set_ylabel('Non-CL AGN Frequency', color='black')
    ax1.legend(loc='upper right')

    CLAGN_zscore_binsize = (max(CLAGN_zscores) - min(CLAGN_zscores))/50  # 50 bins
    CLAGN_bins_zscore = np.arange(min(CLAGN_zscores), max(CLAGN_zscores) + 2*CLAGN_zscore_binsize, CLAGN_zscore_binsize)
    MAD_zscore_CLAGN = median_abs_deviation(CLAGN_zscores)
    x_start_CLAGN = median_zscore_CLAGN - MAD_zscore_CLAGN
    x_end_CLAGN = median_zscore_CLAGN + MAD_zscore_CLAGN
    counts_CLAGN, bin_edges_CLAGN = np.histogram(CLAGN_zscores, bins=CLAGN_bins_zscore)
    height_CLAGN = max(counts_CLAGN)/2
    
    ax2.hist(CLAGN_zscores, bins=CLAGN_bins_zscore, color='red', edgecolor='black', label='CLAGN')
    ax2.axvline(median_zscore_CLAGN, linewidth=2, linestyle='--', color='darkred', label=f'CLAGN Median = {median_zscore_CLAGN:.2f}')
    ax2.axvline(three_sigma_zscore, linewidth=2, linestyle='--', color='black', label=f'{i/len(CLAGN_zscores)*100:.1f}% > Threshold = {three_sigma_zscore:.2f}')
    ax2.plot((x_start_CLAGN, x_end_CLAGN), (height_CLAGN, height_CLAGN), linewidth=2, color='black', label = f'Median Absolute Deviation = {MAD_zscore_CLAGN:.2f}')
    ax2.set_xlabel('Z-Score')
    ax2.set_ylabel('CLAGN Frequency', color='black')
    ax2.legend(loc='upper right')

    if brightness == 0:
        plt.suptitle(f'Z-Score Distribution (Dim) - CLAGN & Non-CL AGN Sample {my_sample}')
    elif brightness == 1:
        plt.suptitle(f'Z-Score Distribution (Bright) - CLAGN & Non-CL AGN Sample {my_sample}')
    elif brightness == 2:
        plt.suptitle(f'Z-Score Distribution (All) - CLAGN & Non-CL AGN Sample {my_sample}')
    plt.tight_layout()
    plt.show()


if main_MIR_Zs_hist_bright_dim == 1:
    AGN_quantifying_change_data = pd.read_csv(f'AGN_Quantifying_Change_just_MIR_max_uncs_Sample_{my_sample}.csv')
    AGN_quantifying_change_data = AGN_quantifying_change_data[AGN_quantifying_change_data.iloc[:, 27] >= 0.5]
    AGN_z_score_bright = AGN_quantifying_change_data.iloc[:, 17].tolist()
    AGN_z_score_unc_bright = AGN_quantifying_change_data.iloc[:, 18].tolist()
    AGN_quantifying_change_data = pd.read_csv(f'AGN_Quantifying_Change_just_MIR_max_uncs_Sample_{my_sample}.csv')
    AGN_quantifying_change_data = AGN_quantifying_change_data[AGN_quantifying_change_data.iloc[:, 27] < 0.5]
    AGN_z_score_dim = AGN_quantifying_change_data.iloc[:, 17].tolist()
    AGN_z_score_unc_dim = AGN_quantifying_change_data.iloc[:, 18].tolist()

    AGN_z_score_all = AGN_z_score_bright+AGN_z_score_dim
    median_z_score_AGN_bright = np.nanmedian(AGN_z_score_bright)
    median_z_score_AGN_unc_bright = np.nanmedian(AGN_z_score_unc_bright)
    three_sigma_z_score_bright = median_z_score_AGN_bright + 3*median_z_score_AGN_unc_bright
    median_z_score_AGN_dim = np.nanmedian(AGN_z_score_dim)
    median_z_score_AGN_unc_dim = np.nanmedian(AGN_z_score_unc_dim)
    three_sigma_z_score_dim = median_z_score_AGN_dim + 3*median_z_score_AGN_unc_dim

    AGN_flux_diff_binsize = (max(AGN_z_score_all) - min(AGN_z_score_all))/50  # 50 bins
    AGN_bins_flux_diff = np.arange(min(AGN_z_score_all), max(AGN_z_score_all) + AGN_flux_diff_binsize, AGN_flux_diff_binsize)
    x_start_threshold_bright = median_z_score_AGN_bright - 3*median_z_score_AGN_unc_bright
    x_end_threshold_bright = median_z_score_AGN_bright + 3*median_z_score_AGN_unc_bright
    counts_bright, bin_edges = np.histogram(AGN_z_score_bright, bins=AGN_bins_flux_diff)
    height_bright = max(counts_bright)/2

    x_start_threshold_dim = median_z_score_AGN_dim - 3*median_z_score_AGN_unc_dim
    x_end_threshold_dim = median_z_score_AGN_dim + 3*median_z_score_AGN_unc_dim
    counts_dim, bin_edges = np.histogram(AGN_z_score_dim, bins=AGN_bins_flux_diff)
    height_dim = max(counts_dim)/2

    a = 0
    for zs in AGN_z_score_bright:
        if zs > three_sigma_z_score_bright:
            a += 1
    
    b = 0
    for zs in AGN_z_score_dim:
        if zs > three_sigma_z_score_dim:
            b += 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)  
    ax1.hist(AGN_z_score_bright, bins=AGN_bins_flux_diff, color='blue', alpha=0.7, edgecolor='black', label='Bright Non-CL AGN')
    ax1.hist(AGN_z_score_dim, bins=AGN_bins_flux_diff, color='blueviolet', alpha=0.7, edgecolor='black', label='Dim Non-CL AGN')
    ax1.axvline(median_z_score_AGN_bright, linewidth=2, linestyle='--', color='darkblue', label=f'Bright Non-CL AGN Median = {median_z_score_AGN_bright:.2f}')
    ax1.axvline(median_z_score_AGN_dim, linewidth=2, linestyle='--', color='darkblue', label=f'Dim Non-CL AGN Median = {median_z_score_AGN_dim:.2f}')
    ax1.axvline(three_sigma_z_score_bright, linewidth=2, linestyle='--', color='black', label=f'{a/len(AGN_z_score_bright)*100:.1f}% Bright Non-CL AGN > Bright Threshold = {three_sigma_z_score_bright:.2f}')
    ax1.axvline(three_sigma_z_score_dim, linewidth=2, linestyle='--', color='grey', label=f'{b/len(AGN_z_score_dim)*100:.1f}% Dim Non-CL AGN > Dim Threshold = {three_sigma_z_score_dim:.2f}')
    ax1.plot((x_start_threshold_bright, x_end_threshold_bright), (height_bright+0.75, height_bright+0.75), linewidth=2, color='tan', label = f'3X Median Bright Uncertainty = {3*median_z_score_AGN_unc_bright:.2f}')
    ax1.plot((x_start_threshold_dim, x_end_threshold_dim), (height_dim+0.25, height_dim+0.25), linewidth=2, color='darkorange', label = f'3X Median Dim Uncertainty = {3*median_z_score_AGN_unc_dim:.2f}')
    ax1.set_ylabel('Non-CL AGN Frequency', color='black')
    ax1.legend(loc='upper right')

    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 27] >= 0.5]
    CLAGN_z_score_bright = CLAGN_quantifying_change_data.iloc[:, 17].tolist()
    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 27] < 0.5]
    CLAGN_z_score_dim = CLAGN_quantifying_change_data.iloc[:, 17].tolist()

    CLAGN_z_score_all = CLAGN_z_score_bright+CLAGN_z_score_dim
    median_z_score_CLAGN_bright = np.nanmedian(CLAGN_z_score_bright)
    median_z_score_CLAGN_dim = np.nanmedian(CLAGN_z_score_dim)

    CLAGN_flux_diff_binsize = (max(CLAGN_z_score_all) - min(CLAGN_z_score_all))/50  # 50 bins
    CLAGN_bins_flux_diff = np.arange(min(CLAGN_z_score_all), max(CLAGN_z_score_all) + 2*CLAGN_flux_diff_binsize, CLAGN_flux_diff_binsize)
    
    c = 0
    for zs in CLAGN_z_score_bright:
        if zs > three_sigma_z_score_bright:
            c += 1
    
    d = 0
    for zs in CLAGN_z_score_dim:
        if zs > three_sigma_z_score_dim:
            d += 1

    ax2.hist(CLAGN_z_score_bright, bins=CLAGN_bins_flux_diff, color='brown', alpha=0.8, edgecolor='black', label='Bright CLAGN')
    ax2.hist(CLAGN_z_score_dim, bins=CLAGN_bins_flux_diff, color='salmon', alpha=0.4, edgecolor='black', label='Dim CLAGN')
    ax2.axvline(median_z_score_CLAGN_bright, linewidth=2, linestyle='--', color='darkred', label=f'Bright CLAGN Median = {median_z_score_CLAGN_bright:.2f}')
    ax2.axvline(median_z_score_CLAGN_dim, linewidth=2, linestyle='--', color='darkred', label=f'Dim CLAGN Median = {median_z_score_CLAGN_dim:.2f}')
    ax2.axvline(three_sigma_z_score_bright, linewidth=2, linestyle='--', color='black', label=f'{c/len(CLAGN_z_score_bright)*100:.1f}% Bright CLAGN > Bright Threshold = {three_sigma_z_score_bright:.2f}')
    ax2.axvline(three_sigma_z_score_dim, linewidth=2, linestyle='--', color='grey', label=f'{d/len(CLAGN_z_score_dim)*100:.1f}% Dim CLAGN > Dim Threshold = {three_sigma_z_score_dim:.2f}')
    ax2.set_xlabel('Z-Score')
    ax2.set_ylabel('CLAGN Frequency', color='black')
    ax2.legend(loc='upper right')

    plt.suptitle(f'Z-Score Distribution - CLAGN & Non-CL AGN Sample {my_sample}')
    plt.tight_layout()
    plt.show()


# # #Creating a 2d plot for UV normalised flux difference & z score:
if UV_MIRZ == 1:
    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_zscores, AGN_norm_flux_diff_UV, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_zscores, CLAGN_norm_flux_diff_UV, s= 100, color='red',  label='CLAGN')
    # plt.errorbar(AGN_zscores, AGN_norm_flux_diff, xerr=AGN_zscore_uncs, yerr=AGN_norm_flux_diff_UV_unc, fmt='o', color='blue', label='Non-CL AGN')
    # plt.errorbar(CLAGN_zscores, CLAGN_norm_flux_diff, xerr=CLAGN_zscore_uncs, yerr=CLAGN_norm_flux_diff_UV_unc, fmt='o', color='red',  label='CLAGN')
    plt.axhline(y=three_sigma_UV_NFD, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2)
    plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
    plt.ylim(0, 1.05*max(CLAGN_norm_flux_diff_UV+AGN_norm_flux_diff_UV))
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xlabel("Z-Score", fontsize = 26)
    plt.ylabel("UV NFD", fontsize = 26)
    plt.title(f"Characterising Variability in AGN (Sample {my_sample})", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    ax = plt.gca()
    plt.text(0.99, 0.25, f'{i/len(CLAGN_zscores)*100:.1f}% CLAGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.99, 0.19, f'{j/len(AGN_zscores)*100:.1f}% AGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.99, 0.37, f'{b/len(CLAGN_norm_flux_diff_UV)*100:.1f}% of CLAGN above UV NFD threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.99, 0.31, f'{d/len(AGN_norm_flux_diff_UV)*100:.1f}% of AGN above UV NFD threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    plt.show()


# # #Creating a 2d plot for UV NFD & MIR NFD:
if UV_MIR_NFD == 1:
    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_norm_flux_diff, AGN_norm_flux_diff_UV, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_norm_flux_diff, CLAGN_norm_flux_diff_UV, s= 100, color='red',  label='CLAGN')
    # plt.errorbar(AGN_zscores, AGN_norm_flux_diff, xerr=AGN_zscore_uncs, yerr=AGN_norm_flux_diff_UV_unc, fmt='o', color='blue', label='Non-CL AGN')
    # plt.errorbar(CLAGN_zscores, CLAGN_norm_flux_diff, xerr=CLAGN_zscore_uncs, yerr=CLAGN_norm_flux_diff_UV_unc, fmt='o', color='red',  label='CLAGN')
    plt.axhline(y=three_sigma_UV_NFD, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.axvline(x=three_sigma_norm_flux_diff, color='black', linestyle='--', linewidth=2)
    plt.xlim(0, 1.05*max(AGN_norm_flux_diff+CLAGN_norm_flux_diff))
    plt.ylim(0, 1.05*max(CLAGN_norm_flux_diff_UV+AGN_norm_flux_diff_UV))
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xlabel("NFD", fontsize = 26)
    plt.ylabel("UV NFD", fontsize = 26)
    plt.title(f"Characterising Variability in AGN (Sample {my_sample})", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    ax = plt.gca()
    plt.text(0.99, 0.25, f'{k/len(CLAGN_norm_flux_diff)*100:.1f}% CLAGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.99, 0.19, f'{l/len(AGN_norm_flux_diff)*100:.1f}% AGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.99, 0.37, f'{b/len(CLAGN_norm_flux_diff_UV)*100:.1f}% of CLAGN above UV NFD threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.99, 0.31, f'{d/len(AGN_norm_flux_diff_UV)*100:.1f}% of AGN above UV NFD threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    plt.show()


if UVZ_MIRZ == 1:
    CLAGN_quantifying_change_data_UV = pd.read_csv('CLAGN_Quantifying_Change_UV1.csv')
    CLAGN_quantifying_change_data_MIR = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')

    CLAGN_UV_names = CLAGN_quantifying_change_data_UV.iloc[:, 0]
    CLAGN_zscores_UV = CLAGN_quantifying_change_data_UV.iloc[:, 17]

    CLAGN_MIR_names = CLAGN_quantifying_change_data_MIR.iloc[:, 0]
    CLAGN_zscores_MIR = CLAGN_quantifying_change_data_MIR.iloc[:, 17]

    CLAGN_name_to_zs_UV = dict(zip(CLAGN_UV_names, CLAGN_zscores_UV))
    CLAGN_name_to_zs_MIR = dict(zip(CLAGN_MIR_names, CLAGN_zscores_MIR))
    all_names = set(CLAGN_UV_names).union(set(CLAGN_MIR_names))

    # Extract matching z-scores for names that exist in both datasets
    matched_names = set(CLAGN_name_to_zs_UV.keys()).intersection(set(CLAGN_name_to_zs_MIR.keys()))
    CLAGN_zscores_UV = [CLAGN_name_to_zs_UV[name] for name in matched_names]
    CLAGN_zscores_MIR = [CLAGN_name_to_zs_MIR[name] for name in matched_names]

    AGN_quantifying_change_data_UV = pd.read_csv(f'AGN_Quantifying_Change_Sample_{my_sample}_UV1.csv')
    AGN_quantifying_change_data_MIR = pd.read_csv(f'AGN_Quantifying_Change_just_MIR_max_uncs_Sample_{my_sample}.csv')

    AGN_UV_names = AGN_quantifying_change_data_UV.iloc[:, 0]
    AGN_zscores_UV = AGN_quantifying_change_data_UV.iloc[:, 17]

    AGN_MIR_names = AGN_quantifying_change_data_MIR.iloc[:, 0]
    AGN_zscores_MIR = AGN_quantifying_change_data_MIR.iloc[:, 17]

    AGN_name_to_zs_UV = dict(zip(AGN_UV_names, AGN_zscores_UV))
    AGN_name_to_zs_MIR = dict(zip(AGN_MIR_names, AGN_zscores_MIR))
    all_names = set(AGN_UV_names).union(set(AGN_MIR_names))

    matched_names = set(AGN_name_to_zs_UV.keys()).intersection(set(AGN_name_to_zs_MIR.keys()))
    AGN_zscores_UV = [AGN_name_to_zs_UV[name] for name in matched_names]
    AGN_zscores_MIR = [AGN_name_to_zs_MIR[name] for name in matched_names]

    x = np.linspace(0, min([np.nanmax(CLAGN_zscores_UV+AGN_zscores_UV), np.nanmax(CLAGN_zscores_MIR+AGN_zscores_MIR)]), 100)

    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_zscores_MIR, AGN_zscores_UV, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_zscores_MIR, CLAGN_zscores_UV, s= 100, color='red',  label='CLAGN')
    plt.plot(x, x, color='black', linestyle='-', label = 'y=x') #add a y=x line
    plt.xlim(0, 1.05*max(CLAGN_zscores_MIR+AGN_zscores_MIR))
    plt.ylim(0, 1.05*max(CLAGN_zscores_UV+AGN_zscores_UV))
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xlabel("MIR Z-Score", fontsize = 26)
    plt.ylabel("UV Z-Score", fontsize = 26)
    plt.title(f"Comparing MIR Z-Score & UV Z-Score (Sample {my_sample})", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if UVNFD_MIRNFD == 1:
    CLAGN_quantifying_change_data_UV = pd.read_csv('CLAGN_Quantifying_Change_UV1.csv')
    CLAGN_zscores_UV = CLAGN_quantifying_change_data_UV.iloc[:, 17].tolist()
    CLAGN_norm_flux_diff_UV = CLAGN_quantifying_change_data_UV.iloc[:, 19].tolist()
    CLAGN_quantifying_change_data_MIR = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_zscores_MIR = CLAGN_quantifying_change_data_MIR.iloc[:, 17].tolist()
    CLAGN_norm_flux_diff_MIR = CLAGN_quantifying_change_data_MIR.iloc[:, 19].tolist()
    AGN_quantifying_change_data_UV = pd.read_csv(f'AGN_Quantifying_Change_Sample_{my_sample}_UV1.csv')
    AGN_zscores_UV = AGN_quantifying_change_data_UV.iloc[:, 17].tolist()
    AGN_norm_flux_diff_UV = AGN_quantifying_change_data_UV.iloc[:, 19].tolist()
    AGN_quantifying_change_data_MIR = pd.read_csv(f'AGN_Quantifying_Change_just_MIR_max_uncs_Sample_{my_sample}.csv')
    AGN_zscores_MIR = AGN_quantifying_change_data_MIR.iloc[:, 17].tolist()
    AGN_norm_flux_diff_MIR = AGN_quantifying_change_data_MIR.iloc[:, 19].tolist()
    
    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_zscores_MIR, AGN_zscores_UV, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_zscores_MIR, CLAGN_zscores_UV, s= 100, color='red',  label='CLAGN')
    plt.axhline(y=three_sigma_norm_flux_diff, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2)
    plt.xlim(0, 1.05*max(CLAGN_zscores_MIR+AGN_zscores_MIR))
    plt.ylim(0, 1.05*max(CLAGN_zscores_UV+AGN_zscores_UV))
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xlabel("MIR Z-Score", fontsize = 26)
    plt.ylabel("UV Z-Score", fontsize = 26)
    plt.title(f"Comparing MIR Z-Score & UV Z-Score (Sample {my_sample})", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    ax = plt.gca()
    if my_sample == 1:
        plt.text(0.99, 0.46, f'{i/len(CLAGN_zscores)*100:.1f}% CLAGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.4, f'{j/len(AGN_zscores)*100:.1f}% AGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.16, 0.95, f'{k/len(CLAGN_norm_flux_diff)*100:.1f}% CLAGN > NFD Threshold', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.16, 0.89, f'{l/len(AGN_norm_flux_diff)*100:.1f}% AGN > NFD Threshold', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    elif my_sample == 2:
        plt.text(0.99, 0.52, f'{i/len(CLAGN_zscores)*100:.1f}% CLAGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.46, f'{j/len(AGN_zscores)*100:.1f}% AGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.64, f'{k/len(CLAGN_norm_flux_diff)*100:.1f}% CLAGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.58, f'{l/len(AGN_norm_flux_diff)*100:.1f}% AGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    elif my_sample == 3:
        plt.text(0.99, 0.68, f'{i/len(CLAGN_zscores)*100:.1f}% CLAGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.62, f'{j/len(AGN_zscores)*100:.1f}% AGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.81, f'{k/len(CLAGN_norm_flux_diff)*100:.1f}% CLAGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.75, f'{l/len(AGN_norm_flux_diff)*100:.1f}% AGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)

    # The default transform specifies that text is in data coords, alternatively, you can specify text in axis coords 
    # (0,0 is lower-left and 1,1 is upper-right).
    plt.show()


# # # #Creating a 2d plot of z score vs 2nd lowest flux:
if zs_W1_low == 1:
    plt.figure(figsize=(12, 7))
    # plt.scatter(AGN_zscores, AGN_W1_low_flux, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_zscores, CLAGN_W1_low_flux, color='red',  label='CLAGN')
    plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2, label = 'Threshold')
    plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
    plt.ylim(0, 1.05*max(CLAGN_W1_low_flux+AGN_W1_low_flux))
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xlabel("Z-Score", fontsize = 26)
    plt.ylabel("W1 Band Second Lowest Flux", fontsize = 26)
    plt.title("Second Lowest W1 Flux vs Z-Score", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if zs_W2_low == 1:
    plt.figure(figsize=(12, 7))
    # plt.scatter(AGN_zscores, AGN_W2_low_flux, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_zscores, CLAGN_W2_low_flux, color='red',  label='CLAGN')
    plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2, label = 'Threshold')
    plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
    plt.ylim(0, 1.05*max(CLAGN_W2_low_flux+AGN_W2_low_flux))
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xlabel("Z-Score", fontsize = 26)
    plt.ylabel("W2 Band Second Lowest Flux", fontsize = 26)
    plt.title("Second Lowest W2 Flux vs Z-Score", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# # #Creating a 2d plot of norm flux diff vs 2nd lowest flux:
if NFD_W1_low == 1:
    plt.figure(figsize=(12, 7))
    # plt.scatter(AGN_norm_flux_diff, AGN_W1_low_flux, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_norm_flux_diff, CLAGN_W1_low_flux, color='red',  label='CLAGN')
    plt.axvline(x=three_sigma_norm_flux_diff, color='black', linestyle='--', linewidth=2, label = 'Threshold')
    plt.xlim(0, 1.05*max(CLAGN_norm_flux_diff+AGN_norm_flux_diff))
    plt.ylim(0, 1.05*max(CLAGN_W1_low_flux+AGN_W1_low_flux))
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xlabel("Normalised Flux Difference", fontsize = 26)
    plt.ylabel("W1 Band Second Lowest Flux", fontsize = 26)
    plt.title("Second Lowest W1 Flux vs Normalised Flux Difference", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if NFD_W2_low == 1:
    plt.figure(figsize=(12, 7))
    # plt.scatter(AGN_norm_flux_diff, AGN_W2_low_flux, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_norm_flux_diff, CLAGN_W2_low_flux, color='red',  label='CLAGN')
    plt.axvline(x=three_sigma_norm_flux_diff, color='black', linestyle='--', linewidth=2, label = 'Threshold')
    plt.xlim(0, 1.05*max(CLAGN_norm_flux_diff+AGN_norm_flux_diff))
    plt.ylim(0, 1.05*max(CLAGN_W2_low_flux+AGN_W2_low_flux))
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xlabel("Normalised Flux Difference", fontsize = 26)
    plt.ylabel("W2 Band Second Lowest Flux", fontsize = 26)
    plt.title("Second Lowest W2 Flux vs Normalised Flux Difference", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# # # #Creating a 2d plot of redshift vs z score:
# plt.figure(figsize=(12, 7))
# plt.scatter(CLAGN_zscores, CLAGN_redshifts, color='red', s=100, label='CLAGN')
# plt.scatter(AGN_zscores, AGN_redshifts, color='blue', label='Non-CL AGN')
# plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2)
# plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
# plt.ylim(0, 1.05*max(CLAGN_redshifts+AGN_redshifts))
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel("Z-Score", fontsize = 24)
# plt.ylabel("Redshift", fontsize = 24)
# plt.title("Redshift vs Z-Score", fontsize = 24)
# plt.legend(loc = 'best', fontsize=22)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# # # #Creating a 2d plot of redshift vs 1/(z score):
# inverse_CLAGN_zscores = [1/z for z in CLAGN_zscores]
# inverse_AGN_zscores = [1/z for z in AGN_zscores]

# #line of best fit:
# fit_params_CLAGN = np.polyfit(inverse_CLAGN_zscores, CLAGN_redshifts, 1)  # Degree 1 for a linear fit
# slope_CLAGN, intercept_CLAGN = fit_params_CLAGN
# y_fit_CLAGN = slope_CLAGN*np.array(inverse_CLAGN_zscores)+intercept_CLAGN
# fit_params_AGN = np.polyfit(inverse_AGN_zscores, AGN_redshifts, 1)  # Degree 1 for a linear fit
# slope_AGN, intercept_AGN = fit_params_AGN
# y_fit_AGN = slope_AGN*np.array(inverse_AGN_zscores)+intercept_AGN
# combined_zscores = inverse_CLAGN_zscores+inverse_AGN_zscores
# combined_redshifts = CLAGN_redshifts+AGN_redshifts
# fit_params_both = np.polyfit(combined_zscores, combined_redshifts, 1)  # Degree 1 for a linear fit
# slope_both, intercept_both = fit_params_both
# y_fit_both = slope_both*np.array(combined_zscores)+intercept_both

# plt.figure(figsize=(12, 7))
# plt.scatter(inverse_CLAGN_zscores, CLAGN_redshifts, color='red', s=100, label='CLAGN')
# plt.scatter(inverse_AGN_zscores, AGN_redshifts, color='blue', label='Non-CL AGN')
# plt.plot(inverse_CLAGN_zscores, y_fit_CLAGN, color="red", label=f"CLAGN: y={slope_CLAGN:.2f}x+{intercept_CLAGN:.2f}")
# plt.plot(inverse_AGN_zscores, y_fit_AGN, color="blue", label=f"AGN: y={slope_AGN:.2f}x+{intercept_AGN:.2f}")
# plt.plot(combined_zscores, y_fit_both, color="black", label=f"Comb: y={slope_both:.2f}x+{intercept_both:.2f}")
# plt.xlim(0, 1.05*max(inverse_CLAGN_zscores+inverse_AGN_zscores))
# plt.ylim(0, 1.05*max(CLAGN_redshifts+AGN_redshifts))
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel(r"$\frac{1}{Z-Score}$", fontsize = 24)
# plt.ylabel("Redshift", fontsize = 24)
# plt.title(r"Redshift vs $\frac{1}{Z-Score}$", fontsize = 24)
# plt.legend(loc = 'best', fontsize=22)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# # # #Creating a 2d plot of NFD vs redshift:
# plt.figure(figsize=(12, 7))
# plt.scatter(CLAGN_redshifts, CLAGN_norm_flux_diff, color='red',  label='CLAGN')
# plt.scatter(AGN_redshifts, AGN_norm_flux_diff, color='blue',  label='Non-CL AGN')
# plt.axvline(x=three_sigma_norm_flux_diff, color='black', linestyle='--', linewidth=2)
# plt.xlim(0, 1.05*max(CLAGN_norm_flux_diff+AGN_norm_flux_diff))
# plt.ylim(0, 1.05*max(CLAGN_redshifts+AGN_redshifts))
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel("Redshift", fontsize = 24)
# plt.ylabel("NFD", fontsize = 24)
# plt.title("NFD vs Redshift", fontsize = 24)
# plt.legend(loc = 'best', fontsize=22)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# # # #Creating a 2d plot of Z-score vs W1 low flux:
# plt.figure(figsize=(12, 7))
# plt.scatter(CLAGN_zscores, CLAGN_W1_low_flux, color='red',  label='CLAGN')
# plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2)
# plt.xlim(0, 1.05*max(CLAGN_zscores))
# plt.ylim(0, 1.05*max(CLAGN_W1_low_flux))
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel("Z-Score", fontsize = 24)
# plt.ylabel("W1 Low Flux", fontsize = 24)
# plt.title("W1 Low Flux vs Z-Score", fontsize = 24)
# plt.legend(loc = 'best', fontsize=22)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# # # #Creating a 2d plot of NFD vs W1 low flux:
# plt.figure(figsize=(12, 7))
# plt.scatter(CLAGN_norm_flux_diff, CLAGN_W1_low_flux, color='red',  label='CLAGN')
# plt.axvline(x=three_sigma_norm_flux_diff, color='black', linestyle='--', linewidth=2)
# plt.xlim(0, 1.05*max(CLAGN_norm_flux_diff))
# plt.ylim(0, 1.05*max(CLAGN_W1_low_flux))
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel("NFD", fontsize = 24)
# plt.ylabel("W1 Low Flux", fontsize = 24)
# plt.title("W1 Low Flux vs NFD", fontsize = 24)
# plt.legend(loc = 'best', fontsize=22)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# # # #Creating a 2d plot of W1 low flux vs Redshift:
# plt.figure(figsize=(12, 7))
# plt.scatter(CLAGN_redshifts, CLAGN_W1_low_flux, s=100, color='red',  label='CLAGN')
# plt.scatter(AGN_redshifts, AGN_W1_low_flux, color='blue',  label='Non-CL AGN')
# plt.xlim(0, 1.05*max(CLAGN_redshifts+AGN_redshifts))
# plt.ylim(0, 1.05*max(CLAGN_W1_low_flux+AGN_W1_low_flux))
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel("Redshift", fontsize = 24)
# plt.ylabel("W1 Low Flux", fontsize = 24)
# plt.title("W1 Low Flux vs Redshift", fontsize = 24)
# plt.legend(loc = 'best', fontsize=22)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# # # # #Creating a 2d plot of redshift vs unc:
# plt.figure(figsize=(12, 7))
# plt.scatter(CLAGN_W1_median_flux_unc, CLAGN_redshifts, color='red',  label='CLAGN')
# plt.scatter(AGN_W1_median_flux_unc, AGN_redshifts, color='blue',  label='Non-CL AGN')
# plt.xlim(0, 1.05*max(CLAGN_W1_median_flux_unc+AGN_W1_median_flux_unc))
# plt.ylim(0, 1.05*max(CLAGN_redshifts+AGN_redshifts))
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel("Median Flux Uncertainty", fontsize = 24)
# plt.ylabel("Redshift", fontsize = 24)
# plt.title("Median Flux Uncertainty vs Redshift", fontsize = 24)
# plt.legend(loc = 'best', fontsize=22)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# # # # #Creating a 2d plot of zscore vs z score unc:
# plt.figure(figsize=(12, 7))
# plt.scatter(AGN_zscores, AGN_zscore_uncs, color='blue',  label='Non-CL AGN')
# plt.scatter(CLAGN_zscores, CLAGN_zscore_uncs, color='red',  label='CLAGN')
# plt.xlim(0, 50)
# plt.ylim(0, 20)
# # plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
# # plt.ylim(0, 1.05*max(CLAGN_zscore_uncs+AGN_zscore_uncs))
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel("Z-Score", fontsize = 24)
# plt.ylabel("Z-Score Uncertainty", fontsize = 24)
# plt.title("Z-Score Uncertainty vs Z-Score", fontsize = 24)
# plt.legend(loc = 'best', fontsize=22)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# # # # #Creating a 2d plot of NFD vs NFD unc:
# plt.figure(figsize=(12, 7))
# plt.scatter(AGN_norm_flux_diff_unc, AGN_norm_flux_diff, color='blue',  label='Non-CL AGN')
# plt.scatter(CLAGN_norm_flux_diff_unc, CLAGN_norm_flux_diff, color='red',  label='CLAGN')
# plt.xlim(0, 1.05*max(CLAGN_norm_flux_diff_unc+AGN_norm_flux_diff_unc))
# plt.ylim(0, 1.05*max(CLAGN_norm_flux_diff+AGN_norm_flux_diff))
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel("NFD Uncertainty", fontsize = 24)
# plt.ylabel("NFD", fontsize = 24)
# plt.title("NFD vs NFD Uncertainty", fontsize = 24)
# plt.legend(loc = 'best', fontsize=22)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


## Creating a plot of W1 NFD vs W2 NFD
if W1_vs_W2_NFD == 1:
    max_W1 = np.nanmax(CLAGN_W1_NFD+AGN_W1_NFD)
    max_W2 = np.nanmax(CLAGN_W2_NFD+AGN_W2_NFD)
    CLAGN_median_W1_NFD = np.nanmedian(CLAGN_W1_NFD)
    AGN_median_W1_NFD = np.nanmedian(AGN_W1_NFD)
    CLAGN_median_W2_NFD = np.nanmedian(CLAGN_W2_NFD)
    AGN_median_W2_NFD = np.nanmedian(AGN_W2_NFD)
    x = np.linspace(0, min([max_W1, max_W2]), 100)
    plt.figure(figsize=(12, 7))
    # plt.scatter(AGN_W1_NFD, AGN_W2_NFD, color='blue',  label='Non-CL AGN')
    plt.scatter(CLAGN_W1_NFD, CLAGN_W2_NFD, s=100, color='red',  label='CLAGN')
    plt.plot(x, x, color='black', linestyle='-', label = 'y=x') #add a y=x line
    plt.xlim(0, 1.05*max_W1)
    plt.ylim(0, 1.05*max_W2)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel("W1 NFD", fontsize = 24)
    plt.ylabel("W2 NFD", fontsize = 24)
    if turn_on_off == 0:
        plt.title("W1 NFD vs W2 NFD (turn-off CLAGN)", fontsize = 24)
    elif turn_on_off == 1:
        plt.title("W1 NFD vs W2 NFD (turn-on CLAGN)", fontsize = 24)
    elif turn_on_off == 2:
        plt.title("W1 NFD vs W2 NFD", fontsize = 24)
    plt.grid(True, linestyle='--', alpha=0.5)
    ax = plt.gca()
    if brightness == 0:
        plt.text(0.99, 0.85, f'CLAGN Median W1 NFD = {CLAGN_median_W1_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.73, f'AGN Median W1 NFD = {AGN_median_W1_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.91, f'CLAGN Median W2 NFD = {CLAGN_median_W2_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.79, f'AGN Median W2 NFD = {AGN_median_W2_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.legend(loc = 'lower right', fontsize=22)
    elif brightness == 1:
        plt.text(0.99, 0.31, f'CLAGN Median W1 NFD = {CLAGN_median_W1_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.19, f'AGN Median W1 NFD = {AGN_median_W1_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.37, f'CLAGN Median W2 NFD = {CLAGN_median_W2_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.25, f'AGN Median W2 NFD = {AGN_median_W2_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.legend(loc = 'upper left', fontsize=22)
    plt.tight_layout()
    plt.show()


# ## Creating a plot of W1 Zscore vs W2 Zscore
if W1_vs_W2_Zs == 1:
    max_W1 = np.nanmax(CLAGN_W1_zscore_mean+AGN_W1_zscore_mean)
    max_W2 = np.nanmax(CLAGN_W2_zscore_mean+AGN_W2_zscore_mean)
    CLAGN_median_W1_zs = np.nanmedian(CLAGN_W1_zscore_mean)
    AGN_median_W1_zs = np.nanmedian(AGN_W1_zscore_mean)
    CLAGN_median_W2_zs = np.nanmedian(CLAGN_W2_zscore_mean)
    AGN_median_W2_zs = np.nanmedian(AGN_W2_zscore_mean)
    x = np.linspace(0, min([max_W1, max_W2]), 100)
    plt.figure(figsize=(12, 7))
    # plt.scatter(AGN_W1_zscore_mean, AGN_W2_zscore_mean, color='blue',  label='Non-CL AGN')
    plt.scatter(CLAGN_W1_zscore_mean, CLAGN_W2_zscore_mean, s=100, color='red',  label='CLAGN')
    plt.plot(x, x, color='black', linestyle='-', label = 'y=x') #add a y=x line
    plt.xlim(0, 1.05*max_W1)
    plt.ylim(0, 1.05*max_W2)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel("W1 Z-score", fontsize = 24)
    plt.ylabel("W2 Z-score", fontsize = 24)
    if turn_on_off == 0:
        plt.title("W1 Z-score vs W2 Z-score (turn-off CLAGN)", fontsize = 24)
    elif turn_on_off == 1:
        plt.title("W1 Z-score vs W2 Z-score (turn-on CLAGN)", fontsize = 24)
    elif turn_on_off == 2:
        plt.title("W1 Z-score vs W2 Z-score", fontsize = 24)
    plt.legend(loc = 'best', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.5)
    ax = plt.gca()
    if brightness == 0:
        plt.text(0.99, 0.67, f'CLAGN Median W1 Z-score = {CLAGN_median_W1_zs:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.55, f'AGN Median W1 Z-score = {AGN_median_W1_zs:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.73, f'CLAGN Median W2 Z-score = {CLAGN_median_W2_zs:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.61, f'AGN Median W2 Z-score = {AGN_median_W2_zs:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.legend(loc = 'lower right', fontsize=22)
    elif brightness == 1:
        plt.text(0.01, 0.85, f'CLAGN Median W1 Z-score = {CLAGN_median_W1_zs:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.01, 0.73, f'AGN Median W1 Z-score = {AGN_median_W1_zs:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.01, 0.91, f'CLAGN Median W2 Z-score = {CLAGN_median_W2_zs:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.01, 0.79, f'AGN Median W2 Z-score = {AGN_median_W2_zs:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        plt.legend(loc = 'lower right', fontsize=22)
    elif brightness == 2:
        plt.text(0.99, 0.67, f'CLAGN Median W1 Z-score = {CLAGN_median_W1_zs:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.55, f'AGN Median W1 Z-score = {AGN_median_W1_zs:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.73, f'CLAGN Median W2 Z-score = {CLAGN_median_W2_zs:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.61, f'AGN Median W2 Z-score = {AGN_median_W2_zs:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.legend(loc = 'lower right', fontsize=22)
    plt.tight_layout()
    plt.show()



if Modified_Dev_plot == 1:
    # combined_mod_dev = AGN_mod_dev_list+CLAGN_mod_dev_list
    # # combined_mod_dev = [x for x in combined_mod_dev if x <= 10]
    # combined_mod_dev = [x for x in combined_mod_dev if x > 10]
    # median_mod_dev = np.median(combined_mod_dev)
    # print(max(combined_mod_dev))
    # mod_dev_binsize = (max(combined_mod_dev)-min(combined_mod_dev))/250 #250 bins
    # bins_mod_dev = np.arange(min(combined_mod_dev), max(combined_mod_dev) + 5*mod_dev_binsize, mod_dev_binsize)
    # plt.figure(figsize=(12,7))
    # plt.hist(combined_mod_dev, bins=bins_mod_dev, color='darkorange', edgecolor='black', label=f'binsize = {mod_dev_binsize:.2f}')
    # plt.axvline(median_mod_dev, linewidth=2, linestyle='--', color='black', label = f'Median = {median_mod_dev:.2f}')
    # plt.xlabel('Flux - Modified Deviation from rest of observations for object')
    # plt.ylabel('Frequency')
    # plt.title(f'Distribution of Modified Deviation Values from {len(combined_mod_dev)} Observations')
    # plt.legend(loc='upper right')
    # plt.show()


    #Separating CLAGN mod_dev and non-CL AGN mod_dev:
    threshold = 15
    #CLAGN
    CLAGN_mod_dev_list = [x for x in CLAGN_mod_dev_list if x <= threshold]
    # CLAGN_mod_dev_list = [x for x in CLAGN_mod_dev_list if x > threshold]
    median_mod_dev = np.median(CLAGN_mod_dev_list)
    print(max(CLAGN_mod_dev_list))
    mod_dev_binsize = (max(CLAGN_mod_dev_list)-min(CLAGN_mod_dev_list))/250 #250 bins
    bins_mod_dev = np.arange(min(CLAGN_mod_dev_list), max(CLAGN_mod_dev_list) + 5*mod_dev_binsize, mod_dev_binsize)
    plt.figure(figsize=(12,7))
    plt.hist(CLAGN_mod_dev_list, bins=bins_mod_dev, color='darkorange', edgecolor='black', label=f'binsize = {mod_dev_binsize:.2f}')
    plt.axvline(median_mod_dev, linewidth=2, linestyle='--', color='black', label = f'Median = {median_mod_dev:.2f}')
    plt.xlabel('Flux - Modified Deviation from rest of observations for object')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of CLAGN Modified Deviation Values > {threshold} from {len(CLAGN_mod_dev_list)} Observations')
    plt.legend(loc='upper right')
    plt.show()


    #Non-CL AGN
    AGN_mod_dev_list = [x for x in AGN_mod_dev_list if x <= threshold]
    # AGN_mod_dev_list = [x for x in AGN_mod_dev_list if x > threshold]
    median_mod_dev = np.median(AGN_mod_dev_list)
    print(max(AGN_mod_dev_list))
    mod_dev_binsize = (max(AGN_mod_dev_list)-min(AGN_mod_dev_list))/250 #250 bins
    bins_mod_dev = np.arange(min(AGN_mod_dev_list), max(AGN_mod_dev_list) + 5*mod_dev_binsize, mod_dev_binsize)
    plt.figure(figsize=(12,7))
    plt.hist(AGN_mod_dev_list, bins=bins_mod_dev, color='darkorange', edgecolor='black', label=f'binsize = {mod_dev_binsize:.2f}')
    plt.axvline(median_mod_dev, linewidth=2, linestyle='--', color='black', label = f'Median = {median_mod_dev:.2f}')
    plt.xlabel('Flux - Modified Deviation from rest of observations for object')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of AGN Modified Deviation Values > {threshold} from {len(AGN_mod_dev_list)} Observations')
    plt.legend(loc='upper right')
    plt.show()


if Log_Modified_Dev_plot == 1:
    #CLAGN
    my_var = len(CLAGN_mod_dev_list)
    print(f'Number of CLAGN Observations = {my_var}')
    CLAGN_mod_dev_list = [abs(x) for x in CLAGN_mod_dev_list if abs(x) > 0]
    print(f'Number of CLAGN Observations = 0: {my_var - len(CLAGN_mod_dev_list)}')
    median_mod_dev = np.median(CLAGN_mod_dev_list)
    bins_mod_dev_CLAGN = np.logspace(np.log10(min(CLAGN_mod_dev_list)), np.log10(max(CLAGN_mod_dev_list)), 50)
    plt.figure(figsize=(12,7))
    plt.hist(CLAGN_mod_dev_list, bins=bins_mod_dev_CLAGN, color='darkorange', edgecolor='black')
    plt.axvline(median_mod_dev, linewidth=2, linestyle='--', color='black', label = f'Median = {median_mod_dev:.2f}')
    plt.xlabel('Modified Deviation')
    plt.ylabel('Frequency')
    plt.xscale('log')  # Set x-axis to log scale
    plt.title(f'Distribution of CLAGN Modified Deviation Values - {len(CLAGN_mod_dev_list)} Observations')
    plt.legend(loc='upper right')
    plt.show()


    #Non-CL AGN
    my_var = len(AGN_mod_dev_list)
    print(f'Number of AGN Observations = {my_var}')
    AGN_mod_dev_list = [abs(x) for x in AGN_mod_dev_list if abs(x) > 0]
    print(f'Number of AGN Observations = 0: {my_var - len(AGN_mod_dev_list)}')
    median_mod_dev = np.median(AGN_mod_dev_list)
    bins_mod_dev_AGN = np.logspace(np.log10(min(AGN_mod_dev_list)), np.log10(max(AGN_mod_dev_list)), 50)
    plt.figure(figsize=(12,7))
    plt.hist(AGN_mod_dev_list, bins=bins_mod_dev_AGN, color='darkorange', edgecolor='black')
    plt.axvline(median_mod_dev, linewidth=2, linestyle='--', color='black', label = f'Median = {median_mod_dev:.2f}')
    plt.xlabel('Modified Deviation')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.title(f'Distribution of AGN Modified Deviation Values - {len(AGN_mod_dev_list)} Observations')
    plt.legend(loc='upper right')
    plt.show()


# creating a 2d plot of W1 NFD vs W1 number of epochs
if epochs_NFD_W1 == 1:
    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_W1_epochs, AGN_W1_NFD, color='blue',  label='Non-CL AGN')
    plt.scatter(CLAGN_W1_epochs, CLAGN_W1_NFD, color='red',  label='CLAGN')
    plt.xlim(0, 1.05*max(AGN_W1_epochs+CLAGN_W1_epochs))
    plt.ylim(0, 1.05*max(AGN_W1_NFD+CLAGN_W1_NFD))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel("W1 Epochs", fontsize = 24)
    plt.ylabel("W1 NFD", fontsize = 24)
    plt.title("W1 NFD vs W1 Epochs", fontsize = 24)
    plt.legend(loc = 'best', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# creating a 2d plot of W2 NFD vs W2 number of epochs
if epochs_NFD_W2 == 1:
    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_W2_epochs, AGN_W2_NFD, color='blue',  label='Non-CL AGN')
    plt.scatter(CLAGN_W2_epochs, CLAGN_W2_NFD, color='red',  label='CLAGN')
    plt.xlim(0, 1.05*max(AGN_W2_epochs+CLAGN_W2_epochs))
    max_W2 = max(CLAGN_W2_NFD+AGN_W2_NFD)
    plt.ylim(0, 1.05*max_W2)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel("W2 Epochs", fontsize = 24)
    plt.ylabel("W2 NFD", fontsize = 24)
    plt.title("W2 NFD vs W2 Epochs", fontsize = 24)
    plt.legend(loc = 'best', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# creating a 2d plot of W1 Z-Score vs W1 number of epochs
if epochs_zs_W1 == 1:
    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_W1_epochs, AGN_W1_zscore_mean, color='blue',  label='Non-CL AGN')
    plt.scatter(CLAGN_W1_epochs, CLAGN_W1_zscore_mean, color='red',  label='CLAGN')
    plt.xlim(0, 1.05*max(AGN_W1_epochs+CLAGN_W1_epochs))
    plt.ylim(0, 1.05*max(AGN_W1_zscore_mean+CLAGN_W1_zscore_mean))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel("W1 Epochs", fontsize = 24)
    plt.ylabel("W1 Z-Score", fontsize = 24)
    plt.title("W1 Z-Score vs W1 Epochs", fontsize = 24)
    plt.legend(loc = 'best', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# creating a 2d plot of W2 Z-Score vs W2 number of epochs
if epochs_zs_W2 == 1:
    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_W2_epochs, AGN_W2_zscore_mean, color='blue',  label='Non-CL AGN')
    plt.scatter(CLAGN_W2_epochs, CLAGN_W2_zscore_mean, color='red',  label='CLAGN')
    plt.xlim(0, 1.05*max(AGN_W2_epochs+CLAGN_W2_epochs))
    max_W2 = max(CLAGN_W2_zscore_mean+AGN_W2_zscore_mean)
    plt.ylim(0, 1.05*max_W2)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel("W2 Epochs", fontsize = 24)
    plt.ylabel("W2 Z-Score", fontsize = 24)
    plt.title("W2 Z-Score vs W2 Epochs", fontsize = 24)
    plt.legend(loc = 'best', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if redshift_dist == 1:
    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 27] >= 0.5]
    CLAGN_names_analysis_bright = CLAGN_quantifying_change_data.iloc[:, 0].tolist()
    CLAGN_redshifts_bright = []
    for object_name in CLAGN_names_analysis_bright:
        object_row = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
        redshift = object_row.iloc[0, 3]
        CLAGN_redshifts_bright.append(redshift)

    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 27] < 0.5]
    CLAGN_names_analysis_dim = CLAGN_quantifying_change_data.iloc[:, 0].tolist()
    CLAGN_redshifts_dim = []
    for object_name in CLAGN_names_analysis_dim:
        object_row = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
        redshift = object_row.iloc[0, 3]
        CLAGN_redshifts_dim.append(redshift)

    # combined_redshifts = AGN_redshifts+CLAGN_redshifts
    combined_redshifts = CLAGN_redshifts_bright+CLAGN_redshifts_dim
    redshift_binsize = (max(combined_redshifts)-min(combined_redshifts))/20 #20 bins
    bins_mod_dev = np.arange(min(combined_redshifts), max(combined_redshifts) + redshift_binsize, redshift_binsize)
    AGN_median_redshift = np.median(AGN_redshifts)
    CLAGN_median_redshift_bright = np.median(CLAGN_redshifts_bright)
    CLAGN_median_redshift_dim = np.median(CLAGN_redshifts_dim)
    plt.figure(figsize=(12,7))
    # plt.hist(AGN_redshifts, bins=bins_mod_dev, color='blue', edgecolor='black', label='Non-CL AGN')
    plt.hist(CLAGN_redshifts_bright, bins=bins_mod_dev, color='red', alpha=0.7, edgecolor='black', label='Bright CLAGN')
    plt.hist(CLAGN_redshifts_dim, bins=bins_mod_dev, color='salmon', alpha=0.7, edgecolor='black', label='Dim CLAGN')
    # plt.axvline(AGN_median_redshift, linewidth=2, linestyle='--', color='darkblue', label = f'Non-CL AGN Median = {AGN_median_redshift:.2f}')
    plt.axvline(CLAGN_median_redshift_bright, linewidth=2, linestyle='--', color='darkred', label = f'Bright CLAGN Median = {CLAGN_median_redshift_bright:.2f}')
    plt.axvline(CLAGN_median_redshift_dim, linewidth=2, linestyle='--', color='darkred', label = f'Dim CLAGN Median = {CLAGN_median_redshift_dim:.2f}')
    plt.xlabel('Redshift')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Redshifts - Dim vs Bright CLAGN')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()