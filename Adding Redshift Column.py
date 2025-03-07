import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

no_duplicates_sample = pd.read_csv('guo23_parent_sample_no_duplicates.csv')

guo_CLAGN = pd.read_csv("Guo23_table4_clagn.csv")
AGN_sample_one = pd.read_csv("AGN_Sample.csv")
AGN_sample_two = pd.read_csv("AGN_Sample_two.csv")
AGN_sample_three = pd.read_csv("AGN_Sample_three.csv")

# CLAGN_df = pd.read_csv("CLAGN_Quantifying_Change_just_MIR_max_uncs.csv")
# AGN_sample_one_df = pd.read_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_1.csv')
# AGN_sample_two_df = pd.read_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_2.csv')
# AGN_sample_three_df = pd.read_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_3.csv')
# column_indexes = {col: idx for idx, col in enumerate(AGN_sample_three_df.columns)}
# print(column_indexes)

# #UV data
# CLAGN_df = pd.read_csv("CLAGN_Quantifying_Change_UV_all.csv")
# AGN_sample_one_df = pd.read_csv('AGN_Quantifying_Change_Sample_1.csv')
# AGN_sample_two_df = pd.read_csv('AGN_Quantifying_Change_Sample_2.csv')
# AGN_sample_three_df = pd.read_csv('AGN_Quantifying_Change_Sample_3.csv')

# # # del CLAGN_df['Redshift']
# # # del AGN_sample_one_df['Redshift']
# # # del AGN_sample_two_df['Redshift']
# # # del AGN_sample_three_df['Redshift']

# CLAGN_names = [object_name for object_name in CLAGN_df.iloc[:, 0] if pd.notna(object_name)]
# AGN_names_one = AGN_sample_one_df.iloc[:, 0]
# AGN_names_two = AGN_sample_two_df.iloc[:, 0]
# AGN_names_three = AGN_sample_three_df.iloc[:, 0]

# CLAGN_redshifts = []
# for object_name in CLAGN_names:
#     object_row = no_duplicates_sample[no_duplicates_sample.iloc[:, 3] == object_name]
#     redshift = object_row.iloc[0, 2]
#     CLAGN_redshifts.append(redshift)
    
# AGN_redshifts_one = []
# for object_name in AGN_names_one:
#     object_data = AGN_sample_one[AGN_sample_one.iloc[:, 3] == object_name]
#     redshift = object_data.iloc[0, 2]
#     AGN_redshifts_one.append(redshift)

# AGN_redshifts_two = []
# for object_name in AGN_names_two:
#     object_data = AGN_sample_two[AGN_sample_two.iloc[:, 3] == object_name]
#     redshift = object_data.iloc[0, 2]
#     AGN_redshifts_two.append(redshift)

# AGN_redshifts_three = []
# for object_name in AGN_names_three:
#     object_data = AGN_sample_three[AGN_sample_three.iloc[:, 3] == object_name]
#     redshift = object_data.iloc[0, 2]
#     AGN_redshifts_three.append(redshift)

# CLAGN_df['Redshift'] = CLAGN_redshifts
# AGN_sample_one_df['Redshift'] = AGN_redshifts_one
# AGN_sample_two_df['Redshift'] = AGN_redshifts_two
# AGN_sample_three_df['Redshift'] = AGN_redshifts_three

# # # #Just MIR
# CLAGN_df.to_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv', index=False)
# AGN_sample_one_df.to_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_1.csv', index=False)
# AGN_sample_two_df.to_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_2.csv', index=False)
# AGN_sample_three_df.to_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_3.csv', index=False)

# #UV:
# CLAGN_df.to_csv('CLAGN_Quantifying_Change_UV_all.csv', index=False)
# AGN_sample_one_df.to_csv('AGN_Quantifying_Change_Sample_1.csv', index=False)
# AGN_sample_two_df.to_csv('AGN_Quantifying_Change_Sample_2.csv', index=False)
# AGN_sample_three_df.to_csv('AGN_Quantifying_Change_Sample_3.csv', index=False)


#Making a plot of the redshift distribution of each sample
CLAGN_names = [object_name for object_name in guo_CLAGN.iloc[:, 0] if pd.notna(object_name)]
CLAGN_redshifts_original = []
for CLAGN_name in CLAGN_names:
    CLAGN_data = no_duplicates_sample[no_duplicates_sample.iloc[:, 3] == CLAGN_name]
    CLAGN_z = CLAGN_data.iloc[0, 2]
    CLAGN_redshifts_original.append(CLAGN_z)
AGN_sample_one_redshifts = AGN_sample_one.iloc[:, 2].tolist()
AGN_sample_two_redshifts = AGN_sample_two.iloc[:, 2].tolist()
AGN_sample_three_redshifts = AGN_sample_three.iloc[:, 2].tolist()
CLAGN_redshifts_median = np.median(CLAGN_redshifts_original)
AGN_redshifts_one_median = np.median(AGN_sample_one_redshifts)
AGN_redshifts_two_median = np.median(AGN_sample_two_redshifts)
AGN_redshifts_three_median = np.median(AGN_sample_three_redshifts)

print(f'CLAGN median redshift = {CLAGN_redshifts_median}')
print(f'Sample 1 median redshift = {AGN_redshifts_one_median}')
print(f'Sample 2 median redshift = {AGN_redshifts_two_median}')
print(f'Sample 3 median redshift = {AGN_redshifts_three_median}')


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


# # #Sample 1
# combined_redshifts = CLAGN_redshifts_original + AGN_sample_one_redshifts
# flux_diff_binsize = (max(combined_redshifts) - min(combined_redshifts))/50  # 50 bins
# bins_flux_diff = np.arange(min(combined_redshifts), max(combined_redshifts) + flux_diff_binsize, flux_diff_binsize)
# fig, ax1 = plt.subplots(figsize=(12,7))
# ax2 = ax1.twinx()  # Create second y-axis
# # Plot Non-CL AGN histogram (left y-axis)
# hist1 = ax1.hist(AGN_sample_one_redshifts, bins=bins_flux_diff, color='blue', edgecolor='black', alpha=0.6, label='Non-CL AGN Control')
# # Plot CLAGN histogram (right y-axis)
# hist2 = ax2.hist(CLAGN_redshifts_original, bins=bins_flux_diff, color='red', edgecolor='black', alpha=0.6, label='Guo CLAGN')
# line1 = ax1.axvline(AGN_redshifts_one_median, linewidth=2, linestyle='--', color='darkblue', label=f'Non-CL AGN Median = {AGN_redshifts_one_median:.2f}')
# line2 = ax2.axvline(CLAGN_redshifts_median, linewidth=2, linestyle=':', color='darkred', label=f'CLAGN Median = {CLAGN_redshifts_median:.2f}')
# ax1.tick_params(axis='both', labelsize=22, length=8, width=2)
# ax2.tick_params(axis='both', labelsize=22, length=8, width=2)
# ax1.set_xlabel('Redshift', fontsize = 22)
# ax1.set_ylabel('Non-CL AGN Frequency', color='blue', fontsize = 22)
# ax2.set_ylabel('CLAGN Frequency', color='red', fontsize = 22)
# # Combine legend handles from both axes
# handles = [hist1[2][0], hist2[2][0], line1, line2]  # Use `hist[2][0]` to get a patch handle
# labels = ['Non-CL AGN Control', 'Guo CLAGN', f'Non-CL AGN Median = {AGN_redshifts_one_median:.2f}', f'CLAGN Median = {CLAGN_redshifts_median:.2f}']
# # Add a single legend
# ax1.legend(handles, labels, loc='upper right', fontsize = 21)
# plt.title('Redshift Distribution - Guo CLAGN & Non-CL AGN Control', fontsize=24)
# plt.tight_layout()
# plt.show()


# #Sample 2
# combined_redshifts = CLAGN_redshifts_original+AGN_sample_two_redshifts
# flux_diff_binsize = (max(combined_redshifts)-min(combined_redshifts))/50 #50 bins
# bins_flux_diff = np.arange(min(combined_redshifts), max(combined_redshifts) + flux_diff_binsize, flux_diff_binsize)
# fig, ax1 = plt.subplots(figsize=(12,7))
# ax2 = ax1.twinx()  # Create second y-axis
# # Plot Non-CL AGN histogram (left y-axis)
# hist1 = ax1.hist(AGN_sample_two_redshifts, bins=bins_flux_diff, color='blue', edgecolor='black', alpha=0.6, label='Non-CL AGN')
# # Plot CLAGN histogram (right y-axis)
# hist2 = ax2.hist(CLAGN_redshifts_original, bins=bins_flux_diff, color='red', edgecolor='black', alpha=0.6, label='CLAGN')
# line1 = ax1.axvline(AGN_redshifts_two_median, linewidth=2, linestyle='--', color='darkblue', label=f'Non-CL AGN Median = {AGN_redshifts_two_median:.2f}')
# line2 = ax2.axvline(CLAGN_redshifts_median, linewidth=2, linestyle='--', color='darkred', label=f'CLAGN Median = {CLAGN_redshifts_median:.2f}')
# ax1.set_xlabel('Redshift')
# ax1.set_ylabel('Non-CL AGN Frequency', color='black')
# ax2.set_ylabel('CLAGN Frequency', color='black')
# # Combine legend handles from both axes
# handles = [hist1[2][0], hist2[2][0], line1, line2]  # Use `hist[2][0]` to get a patch handle
# labels = ['Non-CL AGN', 'CLAGN', f'Non-CL AGN Median = {AGN_redshifts_two_median:.2f}', f'CLAGN Median = {CLAGN_redshifts_median:.2f}']
# # Add a single legend
# ax1.legend(handles, labels, loc='upper right')
# plt.title('Redshift Distribution - CLAGN & Non-CL AGN Sample 2')
# plt.tight_layout()
# plt.show()


# #Sample 3
# combined_redshifts = CLAGN_redshifts_original+AGN_sample_three_redshifts
# flux_diff_binsize = (max(combined_redshifts)-min(combined_redshifts))/50 #50 bins
# bins_flux_diff = np.arange(min(combined_redshifts), max(combined_redshifts) + flux_diff_binsize, flux_diff_binsize)
# fig, ax1 = plt.subplots(figsize=(12,7))
# ax2 = ax1.twinx()  # Create second y-axis
# # Plot Non-CL AGN histogram (left y-axis)
# hist1 = ax1.hist(AGN_sample_three_redshifts, bins=bins_flux_diff, color='blue', edgecolor='black', alpha=0.6, label='Non-CL AGN')
# # Plot CLAGN histogram (right y-axis)
# hist2 = ax2.hist(CLAGN_redshifts_original, bins=bins_flux_diff, color='red', edgecolor='black', alpha=0.6, label='CLAGN')
# line1 = ax1.axvline(AGN_redshifts_three_median, linewidth=2, linestyle='--', color='darkblue', label=f'Non-CL AGN Median = {AGN_redshifts_three_median:.2f}')
# line2 = ax2.axvline(CLAGN_redshifts_median, linewidth=2, linestyle='--', color='darkred', label=f'CLAGN Median = {CLAGN_redshifts_median:.2f}')
# ax1.set_xlabel('Redshift')
# ax1.set_ylabel('Non-CL AGN Frequency', color='black')
# ax2.set_ylabel('CLAGN Frequency', color='black')
# # Combine legend handles from both axes
# handles = [hist1[2][0], hist2[2][0], line1, line2]  # Use `hist[2][0]` to get a patch handle
# labels = ['Non-CL AGN', 'CLAGN', f'Non-CL AGN Median = {AGN_redshifts_three_median:.2f}', f'CLAGN Median = {CLAGN_redshifts_median:.2f}']
# # Add a single legend
# ax1.legend(handles, labels, loc='upper right')
# plt.title('Redshift Distribution - CLAGN & Non-CL AGN Sample 3')
# plt.tight_layout()
# plt.show()


#Combining the plots to the same axes
# #Uncomment below if you want only objects analysed distribution.
# CLAGN_redshifts_original = CLAGN_df.iloc[:, 35].tolist()
# AGN_sample_one_redshifts = AGN_sample_one_df.iloc[:, 37].tolist()
# AGN_sample_two_redshifts = AGN_sample_two_df.iloc[:, 37].tolist()
# AGN_sample_three_redshifts = AGN_sample_three_df.iloc[:, 37].tolist()
# CLAGN_redshifts_median = np.median(CLAGN_redshifts_original)
# AGN_redshifts_one_median = np.median(AGN_sample_one_redshifts)
# AGN_redshifts_two_median = np.median(AGN_sample_two_redshifts)
# AGN_redshifts_three_median = np.median(AGN_sample_three_redshifts)
# print(f'CLAGN median redshift = {CLAGN_redshifts_median}')
# print(f'Sample 1 median redshift = {AGN_redshifts_one_median}')
# print(f'Sample 2 median redshift = {AGN_redshifts_two_median}')
# print(f'Sample 3 median redshift = {AGN_redshifts_three_median}')


combined_redshifts = CLAGN_redshifts_original + AGN_sample_one_redshifts + AGN_sample_two_redshifts + AGN_sample_three_redshifts
flux_diff_binsize = (max(combined_redshifts) - min(combined_redshifts)) / 50  # 50 bins
bins_flux_diff = np.arange(min(combined_redshifts), max(combined_redshifts) + flux_diff_binsize, flux_diff_binsize)
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)  # 3 rows, 1 column
AGN_samples = [AGN_sample_one_redshifts, AGN_sample_two_redshifts, AGN_sample_three_redshifts]
AGN_medians = [AGN_redshifts_one_median, AGN_redshifts_two_median, AGN_redshifts_three_median]
sample_labels = ["Sample 1", "Sample 2", "Sample 3"]
max_y_value = 0
for AGN_data in AGN_samples:
    counts, _ = np.histogram(AGN_data, bins=bins_flux_diff)
    max_y_value = max(max_y_value, max(counts))  # Update max y-value

for i, ax in enumerate(axes):
    ax2 = ax.twinx()  # Create second y-axis
    hist1 = ax.hist(AGN_samples[i], bins=bins_flux_diff, color='blue', edgecolor='black', alpha=0.6, label=f'Non-CL AGN: {sample_labels[i]}')
    hist2 = ax2.hist(CLAGN_redshifts_original, bins=bins_flux_diff, color='red', edgecolor='black', alpha=0.6, label='CLAGN')
    line1 = ax.axvline(AGN_medians[i], linewidth=2, linestyle='--', color='darkblue', label=f'Non-CL AGN Median = {AGN_medians[i]:.2f}')
    line2 = ax2.axvline(CLAGN_redshifts_median, linewidth=2, linestyle='--', color='darkred', label=f'CLAGN Median = {CLAGN_redshifts_median:.2f}')
    ax.set_ylabel(f'Non-CL AGN Frequency', color='blue')
    ax2.set_ylabel(f'CLAGN Frequency', color='red')
    ax.set_ylim(0, max_y_value*1.05)
    # Add a legend for each subplot
    handles = [hist1[2][0], hist2[2][0], line1, line2]
    labels = [f'Non-CL AGN: {sample_labels[i]}', 'CLAGN', f'Non-CL AGN Median = {AGN_medians[i]:.2f}', f'CLAGN Median = {CLAGN_redshifts_median:.2f}']
    ax.legend(handles, labels, loc='upper right')

# Set x-label only on the last subplot
axes[-1].set_xlabel('Redshift')
fig.suptitle('Redshift Distribution - CLAGN & Non-CL AGN Samples 1/2/3')
plt.tight_layout()  # Adjust layout to fit title
plt.show()