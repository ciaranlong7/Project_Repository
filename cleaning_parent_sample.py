import numpy as np
import pandas as pd
from sparcl.client import SparclClient
from requests.exceptions import ConnectTimeout
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# client = SparclClient(connect_timeout=10)

# @retry(stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type((ConnectTimeout, TimeoutError, ConnectionError)))
# def get_primary_spectrum(specid): #some objects have multiple spectra for it in DESI- the best one is the 'primary' spectrum    
#     try:
#         res = client.retrieve_by_specid(specid_list=[int(specid)], include=['specprimary'], dataset_list=['DESI-EDR'])

#         records = res.records

#         if not records: #no spectrum could be found:
#             return 0

#         spec_primary = np.array([records[jj].specprimary for jj in range(len(records))])

#         if not np.any(spec_primary): #no primary spectrum could be found
#             return 0
        
#         return 1

#     except (ConnectTimeout, TimeoutError, ConnectionError) as e:
#         temp_save = guo_parent[guo_parent['keep'] == 1]

#         temp_save = temp_save.drop(columns=['keep'])

#         temp_save.to_csv('temp_sample.csv', index=False)

#         print(f"Connection timeout: {e}")
#         print('Temporary save to temp_sample.csv successful')
#         raise ConnectTimeout
    
# guo_parent = pd.read_csv('guo23_parent_sample_no_duplicates.csv')

# # guo_parent = guo_parent.iloc[:30000, :] #checking first 30k rows
# # guo_parent = guo_parent.iloc[30000:55000, :] #checking rows 30k-55k
# guo_parent = guo_parent.iloc[55000:, :] #checking rows 55k until finish

# # Step 2: Apply the function to the 3rd column (index 2)
# guo_parent['keep'] = guo_parent.iloc[:, 10].apply(get_primary_spectrum)

# # Step 3: Filter rows where the function returns 1
# new_parent = guo_parent[guo_parent['keep'] == 1]

# # Step 4: Remove the 'keep' column
# new_parent = new_parent.drop(columns=['keep'])

# # Step 5: Write the filtered DataFrame to a new CSV file
# # new_parent.to_csv('new_parent_sample.csv', index=False)
# # new_parent.to_csv('new_parent_sample_30k.csv', index=False)
# new_parent.to_csv('new_parent_sample_55k.csv', index=False)


# ## Combining the three data frames created
# guo_parent = pd.read_csv('new_parent_sample.csv')
# guo_parent_thirtyk = pd.read_csv('new_parent_sample_30k.csv')
# guo_parent_fivefivek = pd.read_csv('new_parent_sample_55k.csv')
# combined_df = pd.concat([guo_parent, guo_parent_thirtyk, guo_parent_fivefivek], ignore_index=True)
# combined_df.to_csv('combined_guo_parent_sample.csv', index=False)

# parent_sample = pd.read_csv('combined_guo_parent_sample.csv')
# print(f'Objects in parent sample, before duplicates removed = {len(parent_sample)}')
# columns_to_check = parent_sample.columns[[3, 10]] #checking SDSS name, DESI name
# parent_sample = parent_sample.drop_duplicates(subset=columns_to_check)
# print(f'Objects in parent sample, after duplicates removed = {len(parent_sample)}')

# columns_to_check = parent_sample.columns[[10]]
# different_desi = parent_sample.drop_duplicates(subset=columns_to_check)
# print(len(different_desi)) #different desi contains only different desi names, but some duplicate sdss
# columns_to_check = different_desi.columns[[3]]
# different_sdss = different_desi.drop_duplicates(subset=columns_to_check)
# print(len(different_sdss)) #different sdss contains only different desi and sdss names. Some of the desi names that were in different desi have been removed

# desi_names_desi_df = different_desi.iloc[:, 10].tolist()
# desi_names_sdss_df = different_sdss.iloc[:, 10].tolist()

# ## Checking redshift
# parent_sample = pd.read_csv('combined_guo_parent_sample.csv')
# same_redshift = parent_sample[np.abs(parent_sample.iloc[:, 2] - parent_sample.iloc[:, 9]) <= 0.01]
# different_redshift = parent_sample[np.abs(parent_sample.iloc[:, 2] - parent_sample.iloc[:, 9]) > 0.01]

# print(f'Objects in parent sample with same redshift for SDSS & DESI = {len(same_redshift)}')
# print(f'Objects in parent sample with different redshift for SDSS & DESI = {len(different_redshift)}')

# columns_to_check = parent_sample.columns[[10]] #checking DESI name
# same_redshift = same_redshift.drop_duplicates(subset=columns_to_check)
# print(f'Objects in cleaned sample after duplicates removed = {len(same_redshift)}')

# same_redshift.to_csv('clean_parent_sample.csv', index=False)
# different_redshift.to_csv('outside_redshift_sample.csv', index=False)


# # Checking if any instances of an object having SDSS & DESI redshift being > 0.01 apart actually had a redshift that lined up,
# # but was originally removed in the original removal of duplicates - cutting from 110k to 80k
# outside_redshift = pd.read_csv('outside_redshift_sample.csv')
# print(f'number of objects with different redshift = {len(outside_redshift)}')
# guo_parent = pd.read_csv('guo23_parent_sample.csv')

# names_outside_redshift = outside_redshift.iloc[:, 3]

# # Find all rows in guo_parent where the name (4th column) matches any name in the 4th column of CSV A
# matching_rows = guo_parent[guo_parent.iloc[:, 4].isin(names_outside_redshift)]
# print(f'number of matching rows = {len(matching_rows)}')

# same_redshift = matching_rows[np.abs(matching_rows.iloc[:, 3] - matching_rows.iloc[:, 10]) <= 0.01]
# different_redshift = matching_rows[np.abs(matching_rows.iloc[:, 3] - matching_rows.iloc[:, 10]) > 0.01]

# print(f'Objects recovered with a matching redshift that originally was removed (with duplicates) = {len(same_redshift)}')
# print(f'Objects in matching_rows with different redshift for SDSS & DESI (with duplicates) = {len(different_redshift)}')

# columns_to_check = matching_rows.columns[[11]] #checking DESI name
# same_redshift = same_redshift.drop_duplicates(subset=columns_to_check)
# print(f'Objects recovered with a matching redshift after duplicates removed = {len(same_redshift)}')

# same_redshift = same_redshift.drop(same_redshift.columns[0], axis=1) #get rid of index column

# same_redshift_file = 'clean_parent_sample.csv'

# same_redshift.to_csv(same_redshift_file, mode='a', index=False, header=False) #mode a = append mode

# different_redshift = different_redshift.drop_duplicates(subset=columns_to_check)
# print(f'Objects still with a different redshift after duplicates removed = {len(different_redshift)}')
# different_redshift.to_csv('outside_redshift_sample.csv', index=False) 
# #there now are some instances of an object having its different redshift instance in the different csv
# #and it also has a instance of same redshift in the same redshift csv. But doesn't really matter


# clean_sample = pd.read_csv('clean_parent_sample.csv') #sample after recovered objects are appended.
# print(f'length of clean sample = {len(clean_sample)}')
# columns_to_check = clean_sample.columns[[10]] #checking DESI name
# clean_sample = clean_sample.drop_duplicates(subset=columns_to_check)
# clean_sample.to_csv('clean_parent_sample.csv', index=False) 
# print(f'length of clean sample after dropping duplicates = {len(clean_sample)}')
# #This recovered 8353 objects.


# ##Checking if any Guo CLAGN are in sample
# Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")
# clean_sample = pd.read_csv('clean_parent_sample_no_CLAGN.csv')

# object_names = [object_name for object_name in Guo_table4.iloc[:, 0] if pd.notna(object_name)]

# #Below is a list of CLAGN candidate names.
# object_names = [
#     '091452.90+323347.1', '102012.84+324737.2', '102104.98+440355.5', '111634.91+540138.8',
#     '111938.02+513315.5', '112634.33+511554.5', '120332.06+563100.3', '123557.86+582122.9',
#     '124446.47+591510.8', '124931.53+364816.4', '125646.90+233854.8', '140337.55+043126.2',
#     '141735.11+043954.6', '141841.39+333245.8', '142118.41+505945.2', '143421.56+044137.2',
#     '143528.95+134705.7', '144813.63+080734.2', '150232.97+062337.6', '150754.87+274718.7',
#     '151859.62+061840.5', '152425.40+232814.7', '153644.03+330721.1', '153920.83+020857.2',
#     '153952.21+334930.8', '154025.23+211445.6', '155732.71+402546.3', '160129.75+401959.5',
#     '160451.29+553223.4', '160712.23+151432.0', '160808.47+093715.5', '161235.23+053606.8',
#     '161812.85+294416.6', '163959.17+511930.9', '164621.95+393623.8', '165919.33+304347.0',
#     '170407.13+404747.1', '170624.94+423435.1', '172541.38+322937.8', '205407.92+005400.9',
#     '212216.44-014959.8', '212338.71+004107.0', '213430.72-003906.7', '221932.80+251850.4'
# ]

# AGN_sample = pd.read_csv("AGN_Sample.csv")
# print(f'Objects in cleaned sample before CLAGN removed = {len(clean_sample)}')
# no_CLAGN = clean_sample[~clean_sample.iloc[:, 3].isin(object_names)]
# print(f'Objects in cleaned sample after CLAGN removed = {len(no_CLAGN)}')
# no_CLAGN.to_csv('clean_parent_sample_no_CLAGN.csv', index=False)


# ## Now constructing the sample of 280 AGN:
# guo_CLAGN = pd.read_csv('Guo23_table4_clagn.csv')
# guo_CLAGN = guo_CLAGN.dropna(subset=[guo_CLAGN.columns[0]]) #removing the 8 CLAGN with 2 CL lines
# CLAGN_names = [object_name for object_name in guo_CLAGN.iloc[:, 0] if pd.notna(object_name)]
# parent_sample = pd.read_csv('clean_parent_sample_no_CLAGN.csv')
# no_duplicates_sample = pd.read_csv('guo23_parent_sample_no_duplicates.csv')

# AGN_Sample = []
# for _, CLAGN_row in guo_CLAGN.iterrows():
#     CLAGN_z = CLAGN_row.iloc[3]
#     # Calculates the difference between the every single parent z & this CLAGN's z
#     parent_sample['difference'] = np.abs(parent_sample.iloc[:, 9] - CLAGN_z) #Creates a new 'difference' row in parent_sample df
    
#     # Get the 5 rows from the parent sample with the smallest differences
#     closest_rows = parent_sample.nsmallest(5, 'difference')
#     # print(closest_rows.index)
#     # print(closest_rows)
#     parent_sample = parent_sample.drop(closest_rows.index) #remove these 5 rows from the parent sample so they can't be selected on the next iteration

#     # Append these 5 rows to the output list
#     for _, parent_row in closest_rows.iterrows():
#         AGN_Sample.append(parent_row)
# output_df = pd.DataFrame(AGN_Sample)
# output_df.to_csv('AGN_Sample.csv', index=False)

# AGN_Sample_two = []
# #The 280 AGN in the first AGN_sample have already been removed
# for _, CLAGN_row in guo_CLAGN.iterrows():
#     CLAGN_z = CLAGN_row.iloc[3]
#     parent_sample['difference'] = np.abs(parent_sample.iloc[:, 9] - CLAGN_z)
#     closest_rows = parent_sample.nsmallest(5, 'difference')
#     parent_sample = parent_sample.drop(closest_rows.index)

#     for _, parent_row in closest_rows.iterrows():
#         AGN_Sample_two.append(parent_row)
# output_df_two = pd.DataFrame(AGN_Sample_two)
# output_df_two.to_csv('AGN_Sample_two.csv', index=False)

# AGN_Sample_three = []
# #The 560 AGN in the first & second AGN_sample have already been removed
# for _, CLAGN_row in guo_CLAGN.iterrows():
#     CLAGN_z = CLAGN_row.iloc[3]
#     parent_sample['difference'] = np.abs(parent_sample.iloc[:, 9] - CLAGN_z)
#     closest_rows = parent_sample.nsmallest(5, 'difference')
#     parent_sample = parent_sample.drop(closest_rows.index)

#     for _, parent_row in closest_rows.iterrows():
#         AGN_Sample_three.append(parent_row)
# output_df_three = pd.DataFrame(AGN_Sample_three)
# output_df_three.to_csv('AGN_Sample_three.csv', index=False)


# #Final clean check
# guo_sample = pd.read_csv('guo23_parent_sample.csv')
# parent_sample = pd.read_csv('combined_guo_parent_sample.csv')
# print(f'Objects in parent sample, before desi duplicate names removed = {len(parent_sample)}')
# columns_to_check = parent_sample.columns[[10]] #checking DESI name
# parent_sample = parent_sample.drop_duplicates(subset=columns_to_check)
# print(f'Objects in parent sample, after desi duplicate names removed = {len(parent_sample)}')

# desi_names_to_match = parent_sample.iloc[:, 10]

# print(f'Rows in massive Guo sample: {len(guo_sample)}')
# filtered_guo_sample = guo_sample[guo_sample.iloc[:, 11].isin(desi_names_to_match)]
# print(f'Rows in Guo sample with public desi: {len(filtered_guo_sample)}')

# same_redshift = filtered_guo_sample[np.abs(filtered_guo_sample.iloc[:, 3] - filtered_guo_sample.iloc[:, 10]) <= 0.01]
# different_redshift = filtered_guo_sample[np.abs(filtered_guo_sample.iloc[:, 3] - filtered_guo_sample.iloc[:, 10]) > 0.01]

# print(f'Objects in parent sample with same redshift for SDSS & DESI = {len(same_redshift)}')
# print(f'Objects in parent sample with different redshift for SDSS & DESI = {len(different_redshift)}')

# same_redshift = same_redshift.drop(same_redshift.columns[0], axis=1) #drop old index column
# columns_to_check = same_redshift.columns[[10]] #checking DESI name
# same_redshift = same_redshift.drop_duplicates(subset=columns_to_check)
# print(f'Objects in cleaned sample after desi duplicates removed = {len(same_redshift)}')
# columns_to_check = same_redshift.columns[[3]] #checking sdss name
# same_redshift = same_redshift.drop_duplicates(subset=columns_to_check)
# print(f'Objects in cleaned sample after desi and sdss duplicates removed = {len(same_redshift)}')

# same_redshift.to_csv('clean_parent_sample.csv', index=False)
# different_redshift.to_csv('outside_redshift_sample.csv', index=False)

# Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")
# object_names = [object_name for object_name in Guo_table4.iloc[:, 0] if pd.notna(object_name)]
# no_CLAGN = same_redshift[~same_redshift.iloc[:, 3].isin(object_names)]
# print(f'Objects in cleaned sample after CLAGN removed = {len(no_CLAGN)}')
# no_CLAGN.to_csv('clean_parent_sample_no_CLAGN.csv', index=False)


# AGN_sample = pd.read_csv("AGN_Sample.csv")
# names = AGN_sample.iloc[:, 3]
# AGN_sample_two = pd.read_csv("AGN_Sample_two.csv")
# names_two = AGN_sample.iloc[:, 3]
# AGN_sample_three = pd.read_csv("AGN_Sample_three.csv")
# names_three = AGN_sample.iloc[:, 3]

# AGN_new_sample = output_df
# names_new = AGN_new_sample.iloc[:, 3]
# AGN_new_sample_two = output_df_two
# names_new_two = AGN_new_sample_two.iloc[:, 3]
# AGN_new_sample_three = output_df_three
# names_new_three = AGN_new_sample_three.iloc[:, 3]

# difference = set(names) - set(names_new)
# print(len(difference))
# difference_new = set(names_new) - set(names)
# print(len(difference))
# if difference:
#     print(f"The names in AGN_sample that are not in AGN_new_sample are: {difference}")
# if difference_new:
#     print(f"The names in AGN_new_sample that are not in AGN_sample are: {difference_new}")

# difference_two = set(names_two) - set(names_new_two)
# print(len(difference_two))
# difference_new_two = set(names_new_two) - set(names_two)
# print(len(difference_new_two))
# if difference_two:
#     print(f"The names in AGN_sample_two that are not in AGN_new_sample_two are: {difference_two}")
# if difference_new_two:
#     print(f"The names in AGN_new_sample_two that are not in AGN_sample_two are: {difference_new_two}")

# difference_three = set(names_three) - set(names_new_three)
# print(len(difference_three))
# difference_new_three = set(names_new_three) - set(names_three)
# print(len(difference_new_three))
# if difference_three:
#     print(f"The names in AGN_sample_three that are not in AGN_new_sample_three are: {difference_three}")
# if difference_new_three:
#     print(f"The names in AGN_new_sample_three that are not in AGN_sample_three are: {difference_new_three}")

# sample_difference_two = set(names_new) - set(names_new_two)
# print(len(sample_difference_two))
# sample_difference_three = set(names_new) - set(names_new_three)
# print(len(sample_difference_three))
# sample_difference = set(names_new_two) - set(names_new_three)
# print(len(sample_difference))


##Uncomment below - then check the AGN figures 'extra' folder for any spurious epochs. after that I have my complete AGN sample 1 results
##Run quantifying change code - then I will have all up to date versions of my plots

# # # ## Combining the three data frames created
quantifying_change = pd.read_csv('AGN_Quantifying_Change_Sample_1_UV_all.csv')
print(len(quantifying_change))
# quantifying_change_extra = pd.read_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_1_extra.csv')
# print(len(quantifying_change_extra))
# # quantifying_change_extra_v2 = pd.read_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_2_Extra_v2.csv')
# # print(len(quantifying_change_extra_v2))
# # quantifying_change_extra_v3 = pd.read_csv('AGN_Quantifying_Change_sample_1_extra_v3.csv')
# # print(len(quantifying_change_extra_v3))
# # combined_df = pd.concat([quantifying_change, quantifying_change_extra, quantifying_change_extra_v2, quantifying_change_extra_v3], ignore_index=True)
# # combined_df = pd.concat([quantifying_change, quantifying_change_extra, quantifying_change_extra_v2], ignore_index=True)
# combined_df = pd.concat([quantifying_change, quantifying_change_extra], ignore_index=True)
# combined_df.to_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_1.csv', index=False)

## Don't uncomment below this

# # # Names_to_redo = pd.read_excel('Names_to_redo.xlsx')
# # # Names_to_redo = set(Names_to_redo.iloc[:, 0].tolist())
# quantifying_change = pd.read_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_1.csv')
# print(len(quantifying_change))
# quantifying_change_filtered = quantifying_change[~quantifying_change.iloc[:, 0].isin(Names_to_redo)]
# print(len(quantifying_change_filtered))
# quantifying_change_filtered.to_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_1.csv', index=False)


# quantifying_change = pd.read_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_1.csv')
# quantifying_change = quantifying_change.drop('Redshift', axis=1)
# quantifying_change.to_csv('AGN_Quantifying_Change_just_MIR_max_uncs_Sample_1.csv', index=False)



# ## Checking redshift
# parent_sample = pd.read_csv('guo23_parent_sample.csv')
# parent_sample = parent_sample.drop(parent_sample.columns[0], axis=1) #get rid of index column
# print(f'Objects in parent sample: {len(parent_sample)}')
# same_redshift = parent_sample[np.abs(parent_sample.iloc[:, 2] - parent_sample.iloc[:, 9]) <= 0.01]

# print(f'Objects in parent sample with same redshift for SDSS & DESI = {len(same_redshift)}')

# columns_to_check = parent_sample.columns[[10]] #checking DESI name
# same_redshift = same_redshift.drop_duplicates(subset=columns_to_check)
# print(f'Objects in cleaned sample after DESI duplicates removed = {len(same_redshift)}')

# columns_to_check = parent_sample.columns[[3]] #checking SDSS name
# same_redshift = same_redshift.drop_duplicates(subset=columns_to_check)
# print(f'Objects in cleaned sample after SDSS duplicates removed = {len(same_redshift)}')

# same_redshift.to_csv('guo23_parent_sample_no_duplicates.csv', index=False)

# #Testing that all CLAGN remain in the no_duplicates sample - I find one object isn't (213135.84+001517.0)
# #Redshifts are about 0.015 apart. I just take the DESI redshift and then add this object back into the no duplicates sample.
# Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")
# parent_sample_no_duplicates = pd.read_csv('guo23_parent_sample_no_duplicates.csv')
# parent_sample = pd.read_csv('guo23_parent_sample.csv')
# parent_sample = parent_sample.drop(parent_sample.columns[0], axis=1) #get rid of index column
# object_names = [object_name for object_name in Guo_table4.iloc[:, 0] if pd.notna(object_name)]
# for i, object_name in enumerate(object_names):
#     object_data = parent_sample_no_duplicates[parent_sample_no_duplicates.iloc[:, 3] == object_name]
#     if len(object_data) == 0:
#         object_data = parent_sample[parent_sample.iloc[:, 3] == object_name]
#         object_data.iloc[0, 2] = object_data.iloc[0, 9] #sdss redshift becomes desi redshift.
#         parent_sample_no_duplicates = pd.concat([parent_sample_no_duplicates, object_data], ignore_index=True)
#         parent_sample_no_duplicates.to_csv('guo23_parent_sample_no_duplicates.csv', index=False)


# # # Putting UV analysis from older csv file into new one.
# MIR_interp_df = pd.read_csv("AGN_Quantifying_Change_Sample_1_UV0.csv")
# UV_analysis_df = pd.read_csv("AGN_Quantifying_Change_Sample_1_All_UV1.csv")

# #Renaming columns 21 and 22 so the names are the same in each df
# UV_analysis_df.rename(columns={
#     'Mean UV Flux Change DESI - SDSS': 'Median UV Flux Diff On-Off',
#     'Mean UV Flux Change DESI - SDSS Unc': 'Median UV Flux Diff On-Off Unc'
# }, inplace=True)

# temp_merged_df = pd.merge(MIR_interp_df, UV_analysis_df, on='Object', suffixes=('_df1', '_df2'), how='left')
# MIR_interp_df.iloc[:, 21] = temp_merged_df['Median UV Flux Diff On-Off_df2'].fillna(MIR_interp_df.iloc[:, 21]) #index 21 = UV NFD
# MIR_interp_df.iloc[:, 22] = temp_merged_df['Median UV Flux Diff On-Off Unc_df2'].fillna(MIR_interp_df.iloc[:, 22]) #index 22 = UV NFD unc


# MIR_interp_df.to_csv('AGN_Quantifying_Change_Sample_1_UV1.csv', index=False)