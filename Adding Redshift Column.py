import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")
AGN_sample_one = pd.read_csv("AGN_Sample.csv")
AGN_sample_two = pd.read_csv("AGN_Sample_two.csv")
AGN_sample_three = pd.read_csv("AGN_Sample_three.csv")

CLAGN_df = pd.read_csv("CLAGN_Quantifying_Change_just_MIR_max_uncs.csv")
AGN_sample_one_df = pd.read_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_1.csv')
AGN_sample_two_df = pd.read_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_2.csv')
AGN_sample_three_df = pd.read_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_3.csv')

CLAGN_names = [object_name for object_name in CLAGN_df.iloc[:, 0] if pd.notna(object_name)]
AGN_names_one = AGN_sample_one_df.iloc[:, 0]
AGN_names_two = AGN_sample_two_df.iloc[:, 0]
AGN_names_three = AGN_sample_three_df.iloc[:, 0]

CLAGN_redshifts = []
for object_name in CLAGN_names:
    object_row = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
    redshift = object_row.iloc[0, 3]
    CLAGN_redshifts.append(redshift)

AGN_redshifts_one = []
for object_name in AGN_names_one:
    object_data = AGN_sample_one[AGN_sample_one.iloc[:, 3] == object_name]
    redshift = object_data.iloc[0, 2]
    AGN_redshifts_one.append(redshift)

AGN_redshifts_two = []
for object_name in AGN_names_two:
    object_data = AGN_sample_two[AGN_sample_two.iloc[:, 3] == object_name]
    redshift = object_data.iloc[0, 2]
    AGN_redshifts_two.append(redshift)

AGN_redshifts_three = []
for object_name in AGN_names_three:
    object_data = AGN_sample_three[AGN_sample_three.iloc[:, 3] == object_name]
    redshift = object_data.iloc[0, 2]
    AGN_redshifts_three.append(redshift)

CLAGN_df['Redshift'] = CLAGN_redshifts
AGN_sample_one_df['Redshift'] = AGN_redshifts_one
AGN_sample_two_df['Redshift'] = AGN_redshifts_two
AGN_sample_three_df['Redshift'] = AGN_redshifts_three

# CLAGN_df.to_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv', index=False)
# AGN_sample_one_df.to_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_1.csv', index=False)
# AGN_sample_two_df.to_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_2.csv', index=False)
# AGN_sample_three_df.to_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_3.csv', index=False)


#Making a plot of the redshift distribution of each sample
CLAGN_redshifts_original = [redshift for redshift in Guo_table4.iloc[:, 3] if pd.notna(redshift)]
AGN_sample_one_redshifts = AGN_sample_one.iloc[:, 2].tolist()
AGN_sample_two_redshifts = AGN_sample_two.iloc[:, 2].tolist()
AGN_sample_three_redshifts = AGN_sample_three.iloc[:, 2].tolist()
CLAGN_redshifts_median = np.median(CLAGN_redshifts_original)
AGN_redshifts_one_median = np.median(AGN_redshifts_one)
AGN_redshifts_two_median = np.median(AGN_redshifts_two)
AGN_redshifts_three_median = np.median(AGN_redshifts_three)


# #Making a plot of the redshift distribution for the whole sample
# whole_sample = pd.read_csv("clean_parent_sample_no_CLAGN.csv")
# redshifts_whole_sample = whole_sample.iloc[:, 2].tolist()
# whole_sample_redshifts_median = np.median(redshifts_whole_sample)
# flux_diff_binsize = (max(redshifts_whole_sample)-min(redshifts_whole_sample))/100 #100 bins
# bins_flux_diff = np.arange(min(redshifts_whole_sample), max(redshifts_whole_sample) + flux_diff_binsize, flux_diff_binsize)
# plt.figure(figsize=(12,7))
# plt.hist(redshifts_whole_sample, bins=bins_flux_diff, color='blue', edgecolor='black', label=f'Non-CL AGN')
# plt.hist(CLAGN_redshifts_original, bins=bins_flux_diff, color='red', edgecolor='black', label=f'CLAGN')
# plt.axvline(whole_sample_redshifts_median, linewidth=2, linestyle='--', color='darkblue', label = f'Non-CL AGN Median = {whole_sample_redshifts_median:.2f}')
# plt.axvline(CLAGN_redshifts_median, linewidth=2, linestyle='--', color='darkred', label = f'CLAGN Median = {CLAGN_redshifts_median:.2f}')
# plt.xlabel('Redshift')
# plt.ylabel('Frequency')
# plt.title(f'Redshift Distribution - CLAGN & Non-CL AGN Sample 1')
# plt.legend(loc='upper right')
# plt.show()


# #Sample 1
# combined_redshifts = CLAGN_redshifts_original+AGN_sample_one_redshifts
# flux_diff_binsize = (max(combined_redshifts)-min(combined_redshifts))/50 #50 bins
# bins_flux_diff = np.arange(min(combined_redshifts), max(combined_redshifts) + flux_diff_binsize, flux_diff_binsize)
# plt.figure(figsize=(12,7))
# plt.hist(AGN_sample_one_redshifts, bins=bins_flux_diff, color='blue', edgecolor='black', label=f'Non-CL AGN')
# plt.hist(CLAGN_redshifts_original, bins=bins_flux_diff, color='red', edgecolor='black', label=f'CLAGN')
# plt.axvline(AGN_redshifts_one_median, linewidth=2, linestyle='--', color='darkblue', label = f'Non-CL AGN Median = {AGN_redshifts_one_median:.2f}')
# plt.axvline(CLAGN_redshifts_median, linewidth=2, linestyle='--', color='darkred', label = f'CLAGN Median = {CLAGN_redshifts_median:.2f}')
# plt.xlabel('Redshift')
# plt.ylabel('Frequency')
# plt.title(f'Redshift Distribution - CLAGN & Non-CL AGN Sample 1')
# plt.legend(loc='upper right')
# plt.show()


# #Sample 2
# combined_redshifts = CLAGN_redshifts_original+AGN_sample_two_redshifts
# flux_diff_binsize = (max(combined_redshifts)-min(combined_redshifts))/50 #50 bins
# bins_flux_diff = np.arange(min(combined_redshifts), max(combined_redshifts) + flux_diff_binsize, flux_diff_binsize)
# plt.figure(figsize=(12,7))
# plt.hist(AGN_sample_two_redshifts, bins=bins_flux_diff, color='blue', edgecolor='black', label=f'Non-CL AGN (Sample 1)')
# plt.hist(CLAGN_redshifts_original, bins=bins_flux_diff, color='red', edgecolor='black', label=f'CLAGN')
# plt.axvline(AGN_redshifts_two_median, linewidth=2, linestyle='--', color='darkblue', label = f'Non-CL AGN (Sample 1) Median = {AGN_redshifts_two_median:.2f}')
# plt.axvline(CLAGN_redshifts_median, linewidth=2, linestyle='--', color='darkred', label = f'CLAGN Median = {CLAGN_redshifts_median:.2f}')
# plt.xlabel('Redshift')
# plt.ylabel('Frequency')
# plt.title(f'Redshift Distribution - CLAGN & Non-CL AGN Sample 2')
# plt.legend(loc='upper right')
# plt.show()


# #Sample 3
# combined_redshifts = CLAGN_redshifts_original+AGN_sample_three_redshifts
# flux_diff_binsize = (max(combined_redshifts)-min(combined_redshifts))/50 #50 bins
# bins_flux_diff = np.arange(min(combined_redshifts), max(combined_redshifts) + flux_diff_binsize, flux_diff_binsize)
# plt.figure(figsize=(12,7))
# plt.hist(AGN_sample_three_redshifts, bins=bins_flux_diff, color='blue', edgecolor='black', label=f'Non-CL AGN')
# plt.hist(CLAGN_redshifts_original, bins=bins_flux_diff, color='red', edgecolor='black', label=f'CLAGN')
# plt.axvline(AGN_redshifts_three_median, linewidth=2, linestyle='--', color='darkblue', label = f'Non-CL AGN Median = {AGN_redshifts_three_median:.2f}')
# plt.axvline(CLAGN_redshifts_median, linewidth=2, linestyle='--', color='darkred', label = f'CLAGN Median = {CLAGN_redshifts_median:.2f}')
# plt.xlabel('Redshift')
# plt.ylabel('Frequency')
# plt.title(f'Redshift Distribution - CLAGN & Non-CL AGN Sample 3')
# plt.legend(loc='upper right')
# plt.show()


#Comibing the plots to the same axes
combined_redshifts = CLAGN_redshifts_original+AGN_sample_one_redshifts+AGN_sample_two_redshifts+AGN_sample_three_redshifts
flux_diff_binsize = (max(combined_redshifts)-min(combined_redshifts))/50 #50 bins
bins_flux_diff = np.arange(min(combined_redshifts), max(combined_redshifts) + flux_diff_binsize, flux_diff_binsize)
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)  # 3 rows, 1 column

axes[0].hist(AGN_sample_one_redshifts, bins=bins_flux_diff, color='blue', edgecolor='black', label=f'Non-CL AGN: Sample 1')
axes[0].hist(CLAGN_redshifts_original, bins=bins_flux_diff, color='red', edgecolor='black', label=f'CLAGN')
axes[0].axvline(AGN_redshifts_one_median, linewidth=2, linestyle='--', color='darkblue', label = f'Non-CL AGN Median = {AGN_redshifts_one_median:.2f}')
axes[0].axvline(CLAGN_redshifts_median, linewidth=2, linestyle='--', color='darkred', label = f'CLAGN Median = {CLAGN_redshifts_median:.2f}')
axes[0].set_ylabel('Frequency')
axes[0].legend(loc='upper right')
axes[0].set_title(f'Redshift Distribution - CLAGN & Non-CL AGN Samples 1/2/3')

axes[1].hist(AGN_sample_two_redshifts, bins=bins_flux_diff, color='blue', edgecolor='black', label=f'Non-CL AGN: Sample 2')
axes[1].hist(CLAGN_redshifts_original, bins=bins_flux_diff, color='red', edgecolor='black', label=f'CLAGN')
axes[1].axvline(AGN_redshifts_two_median, linewidth=2, linestyle='--', color='darkblue', label = f'Non-CL AGN Median = {AGN_redshifts_two_median:.2f}')
axes[1].axvline(CLAGN_redshifts_median, linewidth=2, linestyle='--', color='darkred', label = f'CLAGN Median = {CLAGN_redshifts_median:.2f}')
axes[1].set_ylabel('Frequency')
axes[1].legend(loc='upper right')

axes[2].hist(AGN_sample_three_redshifts, bins=bins_flux_diff, color='blue', edgecolor='black', label=f'Non-CL AGN: Sample 3')
axes[2].hist(CLAGN_redshifts_original, bins=bins_flux_diff, color='red', edgecolor='black', label=f'CLAGN')
axes[2].axvline(AGN_redshifts_three_median, linewidth=2, linestyle='--', color='darkblue', label = f'Non-CL AGN Median = {AGN_redshifts_three_median:.2f}')
axes[2].axvline(CLAGN_redshifts_median, linewidth=2, linestyle='--', color='darkred', label = f'CLAGN Median = {CLAGN_redshifts_median:.2f}')
axes[2].set_xlabel('Redshift')
axes[2].set_ylabel('Frequency')
axes[2].legend(loc='upper right')

plt.tight_layout()
plt.show()