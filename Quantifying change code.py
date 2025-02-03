import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")
my_sample = 3 #set which AGN sample you want

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

#Quantifying change data
CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
# #Eliminating objects dimmer than 0.5 in W1 band. Keeps np.nan objects
# # With UV
# CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 31] >= 0.5]
# Just MIR
CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 27] >= 0.5]
#Criteria that must have more than 12 epochs in W1 and W2
# CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[(CLAGN_quantifying_change_data.iloc[:, 25] >= 13) & (CLAGN_quantifying_change_data.iloc[:, 26] >= 13)]
print(f'Number of CLAGN Analysed: {len(CLAGN_quantifying_change_data)}')
CLAGN_zscores = CLAGN_quantifying_change_data.iloc[:, 17].tolist()  # 18th column
CLAGN_zscore_uncs = CLAGN_quantifying_change_data.iloc[:, 18].tolist()
CLAGN_norm_flux_diff = CLAGN_quantifying_change_data.iloc[:, 19].tolist()
CLAGN_norm_flux_diff_unc = CLAGN_quantifying_change_data.iloc[:, 20].tolist()
CLAGN_W1_zscore_max = CLAGN_quantifying_change_data.iloc[:, 1].tolist()
CLAGN_W1_zscore_min = CLAGN_quantifying_change_data.iloc[:, 3].tolist()
CLAGN_W1_zscore_mean = [np.nanmean([abs(zmax), abs(zmin)]) for zmax, zmin in zip(CLAGN_W1_zscore_max, CLAGN_W1_zscore_min)]
CLAGN_W1_NFD = CLAGN_quantifying_change_data.iloc[:, 7].tolist()
CLAGN_W2_zscore_max = CLAGN_quantifying_change_data.iloc[:, 9].tolist()
CLAGN_W2_zscore_min = CLAGN_quantifying_change_data.iloc[:, 11].tolist()
CLAGN_W2_zscore_mean = [np.nanmean([abs(zmax), abs(zmin)]) for zmax, zmin in zip(CLAGN_W2_zscore_max, CLAGN_W2_zscore_min)]
CLAGN_W2_NFD = CLAGN_quantifying_change_data.iloc[:, 15].tolist()


# # With UV
# CLAGN_norm_flux_diff_UV = CLAGN_quantifying_change_data.iloc[:, 21].tolist()
# CLAGN_norm_flux_diff_UV_unc = CLAGN_quantifying_change_data.iloc[:, 22].tolist()
# CLAGN_W1_low_flux = CLAGN_quantifying_change_data.iloc[:, 31].tolist()
# CLAGN_W1_low_flux_unc = CLAGN_quantifying_change_data.iloc[:, 32].tolist()
# CLAGN_W1_high_flux_unc = CLAGN_quantifying_change_data.iloc[:, 33].tolist()
# CLAGN_W1_median_dev_flux = CLAGN_quantifying_change_data.iloc[:, 37].tolist() #median_abs_dev of flux for an object in W1 band
# CLAGN_W2_low_flux = CLAGN_quantifying_change_data.iloc[:, 34].tolist()
# CLAGN_W2_low_flux_unc = CLAGN_quantifying_change_data.iloc[:, 35].tolist()
# CLAGN_W2_high_flux_unc = CLAGN_quantifying_change_data.iloc[:, 36].tolist()
# CLAGN_W2_median_dev_flux = CLAGN_quantifying_change_data.iloc[:, 38].tolist()

#Just MIR
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

CLAGN_names_analysis = CLAGN_quantifying_change_data.iloc[:, 0].tolist()

AGN_quantifying_change_data = pd.read_csv(f'AGN_Quantifying_Change_just_MIR_max_uncs_Sample_{my_sample}.csv')
# AGN_quantifying_change_data = pd.read_csv('AGN_Quantifying_Change_just_MIR_max_uncs.csv')
# #Eliminating objects dimmer than 0.5 in W1 band. Keeps np.nan objects
# #With UV
# AGN_quantifying_change_data = AGN_quantifying_change_data[AGN_quantifying_change_data.iloc[:, 31] >= 0.5]
#Just MIR
AGN_quantifying_change_data = AGN_quantifying_change_data[AGN_quantifying_change_data.iloc[:, 27] >= 0.5]
#Criteria that must have more than 12 epochs in W1 and W2
# AGN_quantifying_change_data = AGN_quantifying_change_data[(AGN_quantifying_change_data.iloc[:, 25] >= 13) & (AGN_quantifying_change_data.iloc[:, 26] >= 13)]
print(f'Number of AGN Analysed: {len(AGN_quantifying_change_data)}')
AGN_zscores = AGN_quantifying_change_data.iloc[:, 17].tolist()  # 18th column
AGN_zscore_uncs = AGN_quantifying_change_data.iloc[:, 18].tolist()
AGN_norm_flux_diff = AGN_quantifying_change_data.iloc[:, 19].tolist()
AGN_norm_flux_diff_unc = AGN_quantifying_change_data.iloc[:, 20].tolist()
AGN_W1_zscore_max = AGN_quantifying_change_data.iloc[:, 1].tolist()
AGN_W1_zscore_min = AGN_quantifying_change_data.iloc[:, 3].tolist()
AGN_W1_zscore_mean = [np.nanmean([abs(zmax), abs(zmin)]) for zmax, zmin in zip(AGN_W1_zscore_max, AGN_W1_zscore_min)]
AGN_W1_NFD = AGN_quantifying_change_data.iloc[:, 7].tolist()
AGN_W2_zscore_max = AGN_quantifying_change_data.iloc[:, 9].tolist()
AGN_W2_zscore_min = AGN_quantifying_change_data.iloc[:, 11].tolist()
AGN_W2_zscore_mean = [np.nanmean([abs(zmax), abs(zmin)]) for zmax, zmin in zip(AGN_W2_zscore_max, AGN_W2_zscore_min)]
AGN_W2_NFD = AGN_quantifying_change_data.iloc[:, 15].tolist()

# # With UV
# AGN_norm_flux_diff_UV = AGN_quantifying_change_data.iloc[:, 21].tolist()
# AGN_norm_flux_diff_UV_unc = AGN_quantifying_change_data.iloc[:, 22].tolist()
# AGN_W1_low_flux = AGN_quantifying_change_data.iloc[:, 31].tolist()
# AGN_W1_low_flux_unc = AGN_quantifying_change_data.iloc[:, 32].tolist()
# AGN_W1_high_flux_unc = AGN_quantifying_change_data.iloc[:, 33].tolist()
# AGN_W1_median_dev_flux = AGN_quantifying_change_data.iloc[:, 37].tolist() #median_abs_dev of flux for an object in W1 band
# AGN_W2_low_flux = AGN_quantifying_change_data.iloc[:, 34].tolist()
# AGN_W2_low_flux_unc = AGN_quantifying_change_data.iloc[:, 35].tolist()
# AGN_W2_high_flux_unc = AGN_quantifying_change_data.iloc[:, 36].tolist()
# AGN_W2_median_dev_flux = AGN_quantifying_change_data.iloc[:, 38].tolist()

#Just MIR
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

AGN_names_analysis = AGN_quantifying_change_data.iloc[:, 0].tolist()

# indices = [i for i, num in enumerate(AGN_zscores) if num > 8 and num < 15]
# for index in indices:
#     print(AGN_names_analysis[index])


print(f'CLAGN W1 median 2nd lowest flux = {np.nanmedian(CLAGN_W1_low_flux):.4f}')
print(f'CLAGN W1 median median_abs_dev flux = {np.nanmedian(CLAGN_W1_median_dev_flux):.5f}')
print(f'CLAGN W2 median 2nd lowest flux = {np.nanmedian(CLAGN_W2_low_flux):.4f}')
print(f'CLAGN W2 median median_abs_dev flux = {np.nanmedian(CLAGN_W2_median_dev_flux):.5f}')

print(f'AGN W1 median 2nd lowest flux = {np.nanmedian(AGN_W1_low_flux):.4f}')
print(f'AGN W1 median median_abs_dev flux = {np.nanmedian(AGN_W1_median_dev_flux):.5f}')
print(f'AGN W2 median 2nd lowest flux = {np.nanmedian(AGN_W2_low_flux):.4f}')
print(f'AGN W2 median median_abs_dev flux = {np.nanmedian(AGN_W2_median_dev_flux):.5f}')


CLAGN_redshifts = []
for object_name in CLAGN_names_analysis:
    object_row = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
    redshift = object_row.iloc[0, 3]
    CLAGN_redshifts.append(redshift)

AGN_sample = pd.read_csv('clean_parent_sample_no_CLAGN.csv')
AGN_redshifts = []
for object_name in AGN_names_analysis:
    object_data = AGN_sample[AGN_sample.iloc[:, 3] == object_name]
    if len(object_data) == 0:
        print(object_name)
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

print(f'{i}/{len(CLAGN_zscores)}={i/len(CLAGN_zscores)*100:.3f}% of CLAGN above zscore threshold')
print(f'{k}/{len(CLAGN_norm_flux_diff)}={k/len(CLAGN_norm_flux_diff)*100:.3f}% of CLAGN above norm_diff threshold')
print(f'{j}/{len(AGN_zscores)}={j/len(AGN_zscores)*100:.3f}% of AGN above zscore threshold')
print(f'{l}/{len(AGN_norm_flux_diff)}={l/len(AGN_norm_flux_diff)*100:.3f}% of AGN above norm_diff threshold')


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
plt.figure(figsize=(12, 7))
plt.scatter(AGN_zscores, AGN_norm_flux_diff, color='blue', label='Non-CL AGN')
plt.scatter(CLAGN_zscores, CLAGN_norm_flux_diff, s= 100, color='red',  label='CLAGN')
# plt.errorbar(AGN_zscores, AGN_norm_flux_diff, xerr=AGN_zscore_uncs, yerr=AGN_norm_flux_diff_unc, fmt='o', color='blue', label='Non-CL AGN')
# plt.errorbar(CLAGN_zscores, CLAGN_norm_flux_diff, xerr=CLAGN_zscore_uncs, yerr=CLAGN_norm_flux_diff_unc, fmt='o', color='red',  label='CLAGN')
plt.axhline(y=three_sigma_norm_flux_diff, color='black', linestyle='--', linewidth=2, label='Threshold')
plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2)
# plt.xlim(0, 50)
# plt.ylim(0, 5)
plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
plt.ylim(0, 1.05*max(CLAGN_norm_flux_diff+AGN_norm_flux_diff))
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.xlabel("Z-Score", fontsize = 26)
plt.ylabel("Normalised Flux Difference", fontsize = 26)
plt.title("Characterising MIR Variability in AGN", fontsize = 28)
plt.legend(loc = 'best', fontsize=25)
plt.grid(True, linestyle='--', alpha=0.5)
ax = plt.gca()
plt.tight_layout()
#For median uncs data:
plt.text(0.99, 0.16, f'{i/len(CLAGN_zscores)*100:.1f}% CLAGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
plt.text(0.99, 0.1, f'{j/len(AGN_zscores)*100:.1f}% AGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
plt.text(0.12, 0.9, f'{k/len(CLAGN_norm_flux_diff)*100:.1f}% CLAGN > NFD Threshold', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
plt.text(0.12, 0.84, f'{l/len(AGN_norm_flux_diff)*100:.1f}% AGN > NFD Threshold', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
# The default transform specifies that text is in data coords, alternatively, you can specify text in axis coords 
# (0,0 is lower-left and 1,1 is upper-right).
plt.show()


# # # #Creating a 2d plot of z score vs 2nd lowest flux:
# plt.figure(figsize=(12, 7))
# plt.scatter(AGN_zscores, AGN_W1_low_flux, color='blue', label='Non-CL AGN')
# plt.scatter(CLAGN_zscores, CLAGN_W1_low_flux, color='red',  label='CLAGN')
# plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2, label = 'Threshold')
# plt.xlim(0, 50)
# # plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
# plt.ylim(0, 1.05*max(CLAGN_W1_low_flux+AGN_W1_low_flux))
# plt.xticks(fontsize=26)
# plt.yticks(fontsize=26)
# plt.xlabel("Z-Score", fontsize = 26)
# plt.ylabel("W1 Band Second Lowest Flux", fontsize = 26)
# plt.title("Second Lowest W1 Flux vs Z-Score", fontsize = 28)
# plt.legend(loc = 'best', fontsize=25)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# # # #Creating a 2d plot of norm flux diff vs 2nd lowest flux:
# plt.figure(figsize=(12, 7))
# plt.scatter(AGN_norm_flux_diff, AGN_W1_low_flux, color='blue', label='Non-CL AGN')
# plt.scatter(CLAGN_norm_flux_diff, CLAGN_W1_low_flux, color='red',  label='CLAGN')
# plt.axvline(x=three_sigma_norm_flux_diff, color='black', linestyle='--', linewidth=2, label = 'Threshold')
# plt.xlim(0, 1.05*max(CLAGN_norm_flux_diff+AGN_norm_flux_diff))
# plt.ylim(0, 1.05*max(CLAGN_W1_low_flux+AGN_W1_low_flux))
# plt.xticks(fontsize=26)
# plt.yticks(fontsize=26)
# plt.xlabel("Normalised Flux Difference", fontsize = 26)
# plt.ylabel("W1 Band Second Lowest Flux", fontsize = 26)
# plt.title("Second Lowest W1 Flux vs Normalised Flux Difference", fontsize = 28)
# plt.legend(loc = 'best', fontsize=25)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


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
# plt.scatter(CLAGN_norm_flux_diff, CLAGN_redshifts, color='red',  label='CLAGN')
# plt.scatter(AGN_norm_flux_diff, AGN_redshifts, color='blue',  label='Non-CL AGN')
# plt.axvline(x=three_sigma_norm_flux_diff, color='black', linestyle='--', linewidth=2)
# plt.xlim(0, 1.05*max(CLAGN_norm_flux_diff+AGN_norm_flux_diff))
# plt.ylim(0, 1.05*max(CLAGN_redshifts+AGN_redshifts))
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel("NFD", fontsize = 24)
# plt.ylabel("Redshift", fontsize = 24)
# plt.title("Redshift vs NFD", fontsize = 24)
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


# ## Creating a plot of W1 NFD vs W2 NFD
# plt.figure(figsize=(12, 7))
# plt.scatter(CLAGN_W1_NFD, CLAGN_W2_NFD, s=100, color='red',  label='CLAGN')
# plt.scatter(AGN_W1_NFD, AGN_W2_NFD, color='blue',  label='Non-CL AGN')
# plt.xlim(0, 1.05*max(CLAGN_W1_NFD+AGN_W1_NFD))
# plt.ylim(0, 1.05*max(CLAGN_W2_NFD+AGN_W2_NFD))
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel("W1 NFD", fontsize = 24)
# plt.ylabel("W2 NFD", fontsize = 24)
# plt.title("W1 NFD vs W2 NFD", fontsize = 24)
# plt.legend(loc = 'best', fontsize=22)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# # ## Creating a plot of W1 Zscore vs W2 Zscore
# plt.figure(figsize=(12, 7))
# plt.scatter(CLAGN_W1_zscore_mean, CLAGN_W2_zscore_mean, s=100, color='red',  label='CLAGN')
# plt.scatter(AGN_W1_zscore_mean, AGN_W2_zscore_mean, color='blue',  label='Non-CL AGN')
# plt.xlim(0, 1.05*max(CLAGN_W1_zscore_mean+AGN_W1_zscore_mean))
# plt.ylim(0, 1.05*max(CLAGN_W2_zscore_mean+AGN_W2_zscore_mean))
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel("W1 Z-score", fontsize = 24)
# plt.ylabel("W2 Z-score", fontsize = 24)
# plt.title("W1 Z-score vs W2 Z-score", fontsize = 24)
# plt.legend(loc = 'best', fontsize=22)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()