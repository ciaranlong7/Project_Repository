import numpy as np
from scipy.stats import median_abs_deviation
from scipy.stats import ks_2samp
from scipy.stats import anderson_ksamp
from astropy.cosmology import LambdaCDM
import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd

#comoving distance is the "proper" distance between two points in the universe if the expansion were frozen in time today.
#However, the universe is expanding and light spreads over a larger surface area in an expanding universe.
#so I need luminosity distance
H0 = 70  # Hubble constant in km/s/Mpc
Om0 = 0.27  # Matter density
Ode0 = 0.73  # Dark energy density
# assume a flat lambda-CDM cosmological model. Values the same as Lyu et al. 2022 (10.3847/1538-4357/ac5256)

cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)

def luminosity(flux, redshift):
    flux = flux*1e-17 * u.erg / (u.s * u.cm**2 * u.AA)
    luminosity_distance = cosmo.luminosity_distance(redshift).to(u.cm)
    return flux*4*np.pi*luminosity_distance**2

Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")
my_sample = 1 #set which AGN sample you want
brightness = 2 #0: dim only objects. 1: bright only objects. 2: all objects
my_redshift = 3 #0=low. 1=medium. 2=high. 3=don't filter
MIR_UV = 0 #0=UV. 1=MIR only
turn_on_off = 2 #0=turn-off CLAGN. 1=turn-on CLAGN. #2=don't filter
emission_line = 7 #0=H_alpha, 1=H_beta, 2=MG2, 3=C3_, 4=C4, 5=single emission line objects, 6=dual EL objects, 7=no filter

#bright/dim thresholds
bright_dim_W1 = 0.40
bright_dim_W2 = 0.40

#plots:
main_MIR = 0 #1 if want main zscore and NFD plot.
main_MIR_direction = 0 #NFD vs z score with direction of change
main_MIR_direction_on_vs_off = 0 #NFD vs z score with direction of change - on vs off CLAGN
z_score_hist_direction_on_vs_off = 0 #z score with direction of change histogram - on vs off CLAGN
main_MIR_line_split = 0 # main zscore and NFD plot, for CLAGN only and split by emission line.
main_MIR_NFD_hist = 0 #histogram of distribution of NFD for AGN and non-CL AGN
main_MIR_NFD_hist_bright_dim = 0 #histogram of distribution of NFD for both bright and dim AGN and non-CL AGN
main_MIR_Zs_hist = 0 #histogram of distribution of Z-score for AGN and non-CL AGN
main_MIR_Zs_hist_bright_dim = 0 #histogram of distribution of z-score for both bright and dim AGN and non-CL AGN
UV_MIRZ = 1 #plot of UV NFD vs interpolated z score
UV_MIR_NFD = 1 #plot of UV NFD vs interpolated NFD
UVZ_MIRZ = 0 #plot of interpolated z-score vs max/min z-score
UVNFD_MIRZ = 0 #plot of UV NFD vs max/min z-score
UVNFD_MIRNFD = 0 #plot of UV NFD vs MIR max/min NFD
UV_NFD_dist = 0 #plot of the UV NFD distribution for CLAGN vs non-CL AGN
zs_W1_low = 0 #plot of zscore vs W1 low flux
zs_W2_low = 0 #plot of zscore vs W2 low flux
NFD_W1_low = 0 #plot of NFD vs W1 low flux
NFD_W2_low = 0 #plot of NFD vs W2 low flux
W1_vs_W2_NFD = 0 #plot of W1 NFD vs W2 NFD
W1_vs_W2_NFD_direction = 0 #plot of W1 NFD vs W2 NFD with direction
W1_vs_W2_Zs = 0 #plot of W1 Zs vs W2 Zs
W1_vs_W2_Zs_direction = 0 #plot of W1 Zs vs W2 Zs with direction
Modified_Dev_plot = 0 #plot of distribution of modified deviations
Log_Modified_Dev_plot = 0 #same plot as Modified_Dev_plot but with a log scale
Modified_Dev_epochs_plot = 0 #plot of distribution of modified deviations for epochs
Modified_Dev_vs_epoch_measurements_plot = 0 #plot of modified deviations for epochs vs epoch measurements
Mean_unc_vs_epoch_meas_results = 0 #results of whether a mean unc was used vs number of epoch measurements.
epochs_NFD_W1 = 0 #W1 NFD vs W1 epochs
epochs_NFD_W2 = 0 #W2 NFD vs W2 epochs
epochs_zs_W1 = 0 #W1 Zs vs W1 epochs
epochs_zs_W2 = 0 #W2 Zs vs W2 epochs
redshift_dist_bright_dim = 0 #hist of redshift distribution for objects analysed
redshift_dist_CLAGN_vs_non_CLAGN = 0 #hist of redshift distribution for objects analysed
luminosity_dist_CLAGN = 0 #hist of luminosity distribution for CLAGN analysed
luminosity_dist_AGN = 0 #hist of luminosity distribution for Non-CL AGN analysed
UV_NFD_redshift = 0 #plot of UV NFD vs redshift
UV_NFD_BEL = 0 #plot of UV NFD vs BEL

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

# Fill NaN values in the first column with the value from the row above
Guo_table4_filled = Guo_table4
Guo_table4_filled.iloc[:, 0] = Guo_table4.iloc[:, 0].ffill()

#Quantifying change data - With UV
if MIR_UV == 0:
    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_UV1.csv')
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data.dropna(subset=[CLAGN_quantifying_change_data.columns[21]])

    #filter for only turn-on or turn-off CLAGN:
    if turn_on_off == 0:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4[Guo_table4['transition'] == 'turn-off'].iloc[:, 0])]
    elif turn_on_off == 1:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4[Guo_table4['transition'] == 'turn-on'].iloc[:, 0])]
    
    #Filter by emission line
    if emission_line == 0:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'Halpha'].iloc[:, 0])]
    elif emission_line == 1:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'Hbeta'].iloc[:, 0])]
    elif emission_line == 2:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'Mg ii'].iloc[:, 0])]
    elif emission_line == 3:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'C iii]'].iloc[:, 0])]
    elif emission_line == 4:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'C iv'].iloc[:, 0])]
    elif emission_line == 5:
        name_counts = Guo_table4_filled.iloc[:, 0].value_counts()
        # Get the names of objects with 1 CL emission line
        names_once = name_counts[name_counts == 1].index
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(names_once)]
    elif emission_line == 6:
        name_counts = Guo_table4_filled.iloc[:, 0].value_counts()
        # Get the names of objects with 2 CL emission lines
        names_twice = name_counts[name_counts == 2].index
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(names_twice)]

    # #Drop columns where no UV analysis was performed:
    # CLAGN_quantifying_change_data = CLAGN_quantifying_change_data.dropna(subset=[CLAGN_quantifying_change_data.columns[21]])

    if brightness == 1:
        # Only keep bright objects.
        #First checks W1 min flux. If no W1 data, then checks W2 min flux
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[np.where(CLAGN_quantifying_change_data.iloc[:, 31].notna(),  
        CLAGN_quantifying_change_data.iloc[:, 31] >= bright_dim_W1,  
        CLAGN_quantifying_change_data.iloc[:, 34] >= bright_dim_W2)]
    elif brightness == 0:
        # Only keep dim objects
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[np.where(CLAGN_quantifying_change_data.iloc[:, 31].notna(),  
        CLAGN_quantifying_change_data.iloc[:, 31] < bright_dim_W1,  
        CLAGN_quantifying_change_data.iloc[:, 34] < bright_dim_W2)]
    
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
    CLAGN_W1_min_mjd = CLAGN_quantifying_change_data.iloc[:, 41].tolist()
    CLAGN_W1_max_mjd = CLAGN_quantifying_change_data.iloc[:, 42].tolist()
    CLAGN_W2_min_mjd = CLAGN_quantifying_change_data.iloc[:, 43].tolist()
    CLAGN_W2_max_mjd = CLAGN_quantifying_change_data.iloc[:, 44].tolist()

#Quantifying change data - Just MIR
elif MIR_UV == 1:
    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    
    if turn_on_off == 0:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4[Guo_table4['transition'] == 'turn-off'].iloc[:, 0])]
    elif turn_on_off == 1:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4[Guo_table4['transition'] == 'turn-on'].iloc[:, 0])]
    
    #Filter by emission line
    if emission_line == 0:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'Halpha'].iloc[:, 0])]
    elif emission_line == 1:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'Hbeta'].iloc[:, 0])]
    elif emission_line == 2:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'Mg ii'].iloc[:, 0])]
    elif emission_line == 3:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'C iii]'].iloc[:, 0])]
    elif emission_line == 4:
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'C iv'].iloc[:, 0])]
    elif emission_line == 5:
        name_counts = Guo_table4_filled.iloc[:, 0].value_counts()
        # Get the names of objects with 1 CL emission line
        names_once = name_counts[name_counts == 1].index
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(names_once)]
    elif emission_line == 6:
        name_counts = Guo_table4_filled.iloc[:, 0].value_counts()
        # Get the names of objects with 2 CL emission lines
        names_twice = name_counts[name_counts == 2].index
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(names_twice)]

    if brightness == 1:
        # Only keep bright objects.
        #First checks W1 min flux. If no W1 data, then checks W2 min flux
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[np.where(CLAGN_quantifying_change_data.iloc[:, 27].notna(),  
        CLAGN_quantifying_change_data.iloc[:, 27] >= bright_dim_W1,  
        CLAGN_quantifying_change_data.iloc[:, 30] >= bright_dim_W2)]
    elif brightness == 0:
        # Only keep dim objects
        CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[np.where(CLAGN_quantifying_change_data.iloc[:, 27].notna(),  
        CLAGN_quantifying_change_data.iloc[:, 27] < bright_dim_W1,  
        CLAGN_quantifying_change_data.iloc[:, 30] < bright_dim_W2)]
    
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
    CLAGN_W1_min_mjd = CLAGN_quantifying_change_data.iloc[:, 37].tolist()
    CLAGN_W1_max_mjd = CLAGN_quantifying_change_data.iloc[:, 38].tolist()
    CLAGN_W2_min_mjd = CLAGN_quantifying_change_data.iloc[:, 39].tolist()
    CLAGN_W2_max_mjd = CLAGN_quantifying_change_data.iloc[:, 40].tolist()

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
    AGN_quantifying_change_data = AGN_quantifying_change_data.dropna(subset=[AGN_quantifying_change_data.columns[21]])

    #Drop columns where no UV analysis was performed:
    AGN_quantifying_change_data = AGN_quantifying_change_data.dropna(subset=[AGN_quantifying_change_data.columns[21]])

    if brightness == 1:
        # Only keep bright objects.
        #First checks W1 min flux. If no W1 data, then checks W2 min flux
        AGN_quantifying_change_data = AGN_quantifying_change_data[np.where(AGN_quantifying_change_data.iloc[:, 31].notna(),  
        AGN_quantifying_change_data.iloc[:, 31] >= bright_dim_W1,  
        AGN_quantifying_change_data.iloc[:, 34] >= bright_dim_W2)]
    elif brightness == 0:
        # Only keep dim objects
        AGN_quantifying_change_data = AGN_quantifying_change_data[np.where(AGN_quantifying_change_data.iloc[:, 31].notna(),  
        AGN_quantifying_change_data.iloc[:, 31] < bright_dim_W1,  
        AGN_quantifying_change_data.iloc[:, 34] < bright_dim_W2)]

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
    AGN_W1_min_mjd = AGN_quantifying_change_data.iloc[:, 41].tolist()
    AGN_W1_max_mjd = AGN_quantifying_change_data.iloc[:, 42].tolist()
    AGN_W2_min_mjd = AGN_quantifying_change_data.iloc[:, 43].tolist()
    AGN_W2_max_mjd = AGN_quantifying_change_data.iloc[:, 44].tolist()

#Quantifying change data - Just MIR
elif MIR_UV == 1:
    AGN_quantifying_change_data = pd.read_csv(f'AGN_Quantifying_Change_just_MIR_max_uncs_Sample_{my_sample}.csv')

    if brightness == 1:
        # Only keep bright objects.
        #First checks W1 min flux. If no W1 data, then checks W2 min flux
        AGN_quantifying_change_data = AGN_quantifying_change_data[np.where(AGN_quantifying_change_data.iloc[:, 27].notna(),  
        AGN_quantifying_change_data.iloc[:, 27] >= bright_dim_W1,  
        AGN_quantifying_change_data.iloc[:, 30] >= bright_dim_W2)]
    elif brightness == 0:
        # Only keep dim objects
        AGN_quantifying_change_data = AGN_quantifying_change_data[np.where(AGN_quantifying_change_data.iloc[:, 27].notna(),  
        AGN_quantifying_change_data.iloc[:, 27] < bright_dim_W1,  
        AGN_quantifying_change_data.iloc[:, 30] < bright_dim_W2)]

    if my_redshift == 0:
        AGN_quantifying_change_data = AGN_quantifying_change_data[AGN_quantifying_change_data.iloc[:, 41] <= 0.9]
    elif my_redshift == 1:
        AGN_quantifying_change_data = AGN_quantifying_change_data[(AGN_quantifying_change_data.iloc[:, 41] > 0.9) & (AGN_quantifying_change_data.iloc[:, 41] > 1.8)]
    elif my_redshift == 2:
        AGN_quantifying_change_data = AGN_quantifying_change_data[AGN_quantifying_change_data.iloc[:, 41] > 1.8]

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
    AGN_W1_min_mjd = AGN_quantifying_change_data.iloc[:, 37].tolist()
    AGN_W1_max_mjd = AGN_quantifying_change_data.iloc[:, 38].tolist()
    AGN_W2_min_mjd = AGN_quantifying_change_data.iloc[:, 39].tolist()
    AGN_W2_max_mjd = AGN_quantifying_change_data.iloc[:, 40].tolist()

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


CLAGN_redshifts_analysis = []
for object_name in CLAGN_names_analysis:
    object_row = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
    redshift = object_row.iloc[0, 3]
    CLAGN_redshifts_analysis.append(redshift)

clean_parent_sample = pd.read_csv('clean_parent_sample_no_CLAGN.csv')
AGN_redshifts_analysis = []
for object_name in AGN_names_analysis:
    object_data = clean_parent_sample[clean_parent_sample.iloc[:, 3] == object_name]
    redshift = object_data.iloc[0, 2]
    AGN_redshifts_analysis.append(redshift)

median_CLAGN_redshift_analysis = np.nanmedian(CLAGN_redshifts_analysis)
median_AGN_redshift_analysis = np.nanmedian(AGN_redshifts_analysis)
print(f'Median CLAGN analysed redshift = {median_CLAGN_redshift_analysis:.3f}')
print(f'Median AGN analysed redshift = {median_AGN_redshift_analysis:.3f}')

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
    plt.axhline(y=three_sigma_norm_flux_diff, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2)
    plt.scatter(AGN_zscores, AGN_norm_flux_diff, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_zscores, CLAGN_norm_flux_diff, s=100, color='red',  label='CLAGN')
    # plt.errorbar(AGN_zscores, AGN_norm_flux_diff, xerr=AGN_zscore_uncs, yerr=AGN_norm_flux_diff_unc, fmt='o', color='blue', label='Non-CL AGN')
    # plt.errorbar(CLAGN_zscores, CLAGN_norm_flux_diff, xerr=CLAGN_zscore_uncs, yerr=CLAGN_norm_flux_diff_unc, fmt='o', color='red',  label='CLAGN')
    plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
    plt.ylim(0, 1.05*max(CLAGN_norm_flux_diff+AGN_norm_flux_diff))
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.xlabel("Z-Score", fontsize = 26)
    plt.ylabel("NFD", fontsize = 26)
    plt.title("Characterising MIR Variability - Dim Objects", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.tight_layout()
    ax = plt.gca()
    if my_sample == 1:
        if brightness == 1:
            plt.text(0.99, 0.58, f'{i/len(CLAGN_zscores)*100:.1f}% CLAGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
            plt.text(0.99, 0.52, f'{j/len(AGN_zscores)*100:.1f}% Non-CL AGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
            plt.text(0.14, 0.83, f'{k/len(CLAGN_norm_flux_diff)*100:.1f}% CLAGN > NFD Threshold', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
            plt.text(0.14, 0.77, f'{l/len(AGN_norm_flux_diff)*100:.1f}% Non-CL AGN > NFD Threshold', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        else:
            plt.text(0.99, 0.84, f'{i/len(CLAGN_zscores)*100:.1f}% CLAGN,', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
            plt.text(0.99, 0.78, f'{j/len(AGN_zscores)*100:.1f}% Non-CL AGN', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
            plt.text(0.99, 0.72, f'> Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
            plt.text(0.02, 0.84, f'{k/len(CLAGN_norm_flux_diff)*100:.1f}% CLAGN,', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
            plt.text(0.02, 0.78, f'{l/len(AGN_norm_flux_diff)*100:.1f}% Non-CL AGN', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
            plt.text(0.02, 0.72, f'> NFD Threshold', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    elif my_sample == 2:
        plt.text(0.99, 0.68, f'{i/len(CLAGN_zscores)*100:.1f}% CLAGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.62, f'{j/len(AGN_zscores)*100:.1f}% AGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.81, f'{k/len(CLAGN_norm_flux_diff)*100:.1f}% CLAGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.75, f'{l/len(AGN_norm_flux_diff)*100:.1f}% AGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    elif my_sample == 3:
        plt.text(0.99, 0.68, f'{i/len(CLAGN_zscores)*100:.1f}% CLAGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.62, f'{j/len(AGN_zscores)*100:.1f}% AGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.81, f'{k/len(CLAGN_norm_flux_diff)*100:.1f}% CLAGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.75, f'{l/len(AGN_norm_flux_diff)*100:.1f}% AGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)

    # The default transform specifies that text is in data coords, alternatively, you can specify text in axis coords 
    # (0,0 is lower-left and 1,1 is upper-right).
    plt.show()


if main_MIR_direction == 1:
    CLAGN_W1_min_mjd = np.array(CLAGN_W1_min_mjd)
    CLAGN_W1_max_mjd = np.array(CLAGN_W1_max_mjd)
    CLAGN_W1_NFD = np.array(CLAGN_W1_NFD)
    CLAGN_W1_zscore_mean = np.array(CLAGN_W1_zscore_mean)
    # Find indices where min_mjd > max_mjd
    CLAGN_invert_indices_W1 = CLAGN_W1_min_mjd > CLAGN_W1_max_mjd
    # Make corresponding W1_NFD values negative
    CLAGN_W1_NFD[CLAGN_invert_indices_W1] *= -1
    CLAGN_W1_NFD = CLAGN_W1_NFD.tolist()
    CLAGN_W1_zscore_mean[CLAGN_invert_indices_W1] *= -1
    CLAGN_W1_zscore_mean = CLAGN_W1_zscore_mean.tolist()

    CLAGN_W2_min_mjd = np.array(CLAGN_W2_min_mjd)
    CLAGN_W2_max_mjd = np.array(CLAGN_W2_max_mjd)
    CLAGN_W2_NFD = np.array(CLAGN_W2_NFD)
    CLAGN_W2_zscore_mean = np.array(CLAGN_W2_zscore_mean)
    # Find indices where min_mjd > max_mjd
    CLAGN_invert_indices_W2 = CLAGN_W2_min_mjd > CLAGN_W2_max_mjd
    # Make corresponding W2_NFD values negative
    CLAGN_W2_NFD[CLAGN_invert_indices_W2] *= -1
    CLAGN_W2_NFD = CLAGN_W2_NFD.tolist()
    CLAGN_W2_zscore_mean[CLAGN_invert_indices_W2] *= -1
    CLAGN_W2_zscore_mean = CLAGN_W2_zscore_mean.tolist()

    CLAGN_mean_directional_NFD = [np.nanmean([x,y]) for x, y in zip(CLAGN_W1_NFD, CLAGN_W2_NFD)]
    CLAGN_mean_directional_zscore = [np.nanmean([x,y]) for x, y in zip(CLAGN_W1_zscore_mean, CLAGN_W2_zscore_mean)]

    AGN_W1_min_mjd = np.array(AGN_W1_min_mjd)
    AGN_W1_max_mjd = np.array(AGN_W1_max_mjd)
    AGN_W1_NFD = np.array(AGN_W1_NFD)
    AGN_W1_zscore_mean = np.array(AGN_W1_zscore_mean)
    # Find indices where min_mjd > max_mjd
    AGN_invert_indices_W1 = AGN_W1_min_mjd > AGN_W1_max_mjd
    # Make corresponding W1_NFD values negative
    AGN_W1_NFD[AGN_invert_indices_W1] *= -1
    AGN_W1_NFD = AGN_W1_NFD.tolist()
    AGN_W1_zscore_mean[AGN_invert_indices_W1] *= -1
    AGN_W1_zscore_mean = AGN_W1_zscore_mean.tolist()

    AGN_W2_min_mjd = np.array(AGN_W2_min_mjd)
    AGN_W2_max_mjd = np.array(AGN_W2_max_mjd)
    AGN_W2_NFD = np.array(AGN_W2_NFD)
    AGN_W2_zscore_mean = np.array(AGN_W2_zscore_mean)
    # Find indices where min_mjd > max_mjd
    AGN_invert_indices_W2 = AGN_W2_min_mjd > AGN_W2_max_mjd
    # Make corresponding W2_NFD values negative
    AGN_W2_NFD[AGN_invert_indices_W2] *= -1
    AGN_W2_NFD = AGN_W2_NFD.tolist()
    AGN_W2_zscore_mean[AGN_invert_indices_W2] *= -1
    AGN_W2_zscore_mean = AGN_W2_zscore_mean.tolist()

    AGN_mean_directional_NFD = [np.nanmean([x,y]) for x, y in zip(AGN_W1_NFD, AGN_W2_NFD)]
    AGN_mean_directional_zscore = [np.nanmean([x,y]) for x, y in zip(AGN_W1_zscore_mean, AGN_W2_zscore_mean)]

    max_NFD = np.nanmax(CLAGN_mean_directional_NFD+AGN_mean_directional_NFD)
    min_NFD = np.nanmin(CLAGN_mean_directional_NFD+AGN_mean_directional_NFD)
    max_zscore = np.nanmax(CLAGN_mean_directional_zscore+AGN_mean_directional_zscore)
    min_zscore = np.nanmin(CLAGN_mean_directional_zscore+AGN_mean_directional_zscore)

    CLAGN_median_NFD = np.nanmedian(CLAGN_mean_directional_NFD)
    CLAGN_median_zscore = np.nanmedian(CLAGN_mean_directional_zscore)
    AGN_median_NFD = np.nanmedian(AGN_mean_directional_NFD)
    AGN_median_zscore = np.nanmedian(AGN_mean_directional_zscore)

    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_mean_directional_zscore, AGN_mean_directional_NFD, color='blue',  label='Non-CL AGN')
    plt.scatter(CLAGN_mean_directional_zscore, CLAGN_mean_directional_NFD, s=100, color='red',  label='CLAGN')
    plt.axvline(0, color='black', linestyle=':')
    plt.axhline(0, color='black', linestyle=':')
    plt.xlim(1.1*min_zscore, 1.1*max_zscore)
    plt.ylim(1.1*min_NFD, 1.1*max_NFD)
    plt.tick_params(axis='both', labelsize=24, length=8, width=2)
    plt.xlabel("Directional Z-Score", fontsize = 24)
    plt.ylabel("Directional NFD", fontsize = 24)
    if turn_on_off == 0:
        plt.title("NFD vs Z-score (turn-off CLAGN)", fontsize = 24)
    elif turn_on_off == 1:
        plt.title("NFD vs Z-score (turn-on CLAGN)", fontsize = 24)
    elif turn_on_off == 2:
        plt.title("NFD vs Z-score", fontsize = 24)
    ax = plt.gca()
    if brightness == 0:
        plt.text(0.99, 0.85, f'CLAGN Median NFD = {CLAGN_median_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.91, f'CLAGN Median Z-Score = {CLAGN_median_zscore:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.73, f'AGN Median NFD = {AGN_median_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.79, f'AGN Median Z-Score = {AGN_median_zscore:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.legend(loc = 'lower right', fontsize=22)
    elif brightness == 1:
        plt.text(0.99, 0.31, f'CLAGN Median NFD = {CLAGN_median_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.37, f'CLAGN Median Z-Score = {CLAGN_median_zscore:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.19, f'AGN Median NFD = {AGN_median_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.25, f'AGN Median Z-Score = {AGN_median_zscore:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.legend(loc = 'upper left', fontsize=22)
    elif brightness == 2:
        plt.text(0.04, 0.40, f'CLAGN Median NFD = {CLAGN_median_NFD:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.04, 0.46, f'CLAGN Median Z-Score = {CLAGN_median_zscore:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.19, f'AGN Median NFD = {AGN_median_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.25, f'AGN Median Z-Score = {AGN_median_zscore:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.legend(loc = 'best', fontsize=22)
    plt.tight_layout()
    plt.show()


if main_MIR_direction_on_vs_off == 1:
    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_quantifying_change_data_off = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4[Guo_table4['transition'] == 'turn-off'].iloc[:, 0])]
    CLAGN_W1_zscore_max_off = CLAGN_quantifying_change_data_off.iloc[:, 1].tolist()
    CLAGN_W1_zscore_min_off = CLAGN_quantifying_change_data_off.iloc[:, 3].tolist()
    CLAGN_W1_zscore_mean_off = [
        np.nanmean([abs(zmax), abs(zmin)]) 
        if not (np.isnan(zmax) and np.isnan(zmin))  # Check if both are NaN
        else np.nan  # Assign NaN if both are NaN
        for zmax, zmin in zip(CLAGN_W1_zscore_max_off, CLAGN_W1_zscore_min_off)
    ]
    CLAGN_W1_NFD_off = CLAGN_quantifying_change_data_off.iloc[:, 7].tolist()
    CLAGN_W2_zscore_max_off = CLAGN_quantifying_change_data_off.iloc[:, 9].tolist()
    CLAGN_W2_zscore_min_off = CLAGN_quantifying_change_data_off.iloc[:, 11].tolist()
    CLAGN_W2_zscore_mean_off = [
        np.nanmean([abs(zmax), abs(zmin)]) 
        if not (np.isnan(zmax) and np.isnan(zmin))  # Check if both are NaN
        else np.nan  # Assign NaN if both are NaN
        for zmax, zmin in zip(CLAGN_W2_zscore_max_off, CLAGN_W2_zscore_min_off)
    ]
    CLAGN_W2_NFD_off = CLAGN_quantifying_change_data_off.iloc[:, 15].tolist()
    CLAGN_W1_min_mjd_off = CLAGN_quantifying_change_data_off.iloc[:, 37].tolist()
    CLAGN_W1_max_mjd_off = CLAGN_quantifying_change_data_off.iloc[:, 38].tolist()
    CLAGN_W2_min_mjd_off = CLAGN_quantifying_change_data_off.iloc[:, 39].tolist()
    CLAGN_W2_max_mjd_off = CLAGN_quantifying_change_data_off.iloc[:, 40].tolist()
    
    CLAGN_quantifying_change_data_on = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4[Guo_table4['transition'] == 'turn-on'].iloc[:, 0])]
    CLAGN_W1_zscore_max_on = CLAGN_quantifying_change_data_on.iloc[:, 1].tolist()
    CLAGN_W1_zscore_min_on = CLAGN_quantifying_change_data_on.iloc[:, 3].tolist()
    CLAGN_W1_zscore_mean_on = [
        np.nanmean([abs(zmax), abs(zmin)]) 
        if not (np.isnan(zmax) and np.isnan(zmin))  # Check if both are NaN
        else np.nan  # Assign NaN if both are NaN
        for zmax, zmin in zip(CLAGN_W1_zscore_max_on, CLAGN_W1_zscore_min_on)
    ]
    CLAGN_W1_NFD_on = CLAGN_quantifying_change_data_on.iloc[:, 7].tolist()
    CLAGN_W2_zscore_max_on = CLAGN_quantifying_change_data_on.iloc[:, 9].tolist()
    CLAGN_W2_zscore_min_on = CLAGN_quantifying_change_data_on.iloc[:, 11].tolist()
    CLAGN_W2_zscore_mean_on = [
        np.nanmean([abs(zmax), abs(zmin)]) 
        if not (np.isnan(zmax) and np.isnan(zmin))  # Check if both are NaN
        else np.nan  # Assign NaN if both are NaN
        for zmax, zmin in zip(CLAGN_W2_zscore_max_on, CLAGN_W2_zscore_min_on)
    ]
    CLAGN_W2_NFD_on = CLAGN_quantifying_change_data_on.iloc[:, 15].tolist()
    CLAGN_W1_min_mjd_on = CLAGN_quantifying_change_data_on.iloc[:, 37].tolist()
    CLAGN_W1_max_mjd_on = CLAGN_quantifying_change_data_on.iloc[:, 38].tolist()
    CLAGN_W2_min_mjd_on = CLAGN_quantifying_change_data_on.iloc[:, 39].tolist()
    CLAGN_W2_max_mjd_on = CLAGN_quantifying_change_data_on.iloc[:, 40].tolist()


    CLAGN_W1_min_mjd_off = np.array(CLAGN_W1_min_mjd_off)
    CLAGN_W1_max_mjd_off = np.array(CLAGN_W1_max_mjd_off)
    CLAGN_W1_NFD_off = np.array(CLAGN_W1_NFD_off)
    CLAGN_W1_zscore_mean_off = np.array(CLAGN_W1_zscore_mean_off)
    # Find indices where min_mjd > max_mjd
    CLAGN_invert_indices_W1_off = CLAGN_W1_min_mjd_off > CLAGN_W1_max_mjd_off
    # Make corresponding W1_NFD values negative
    CLAGN_W1_NFD_off[CLAGN_invert_indices_W1_off] *= -1
    CLAGN_W1_NFD_off = CLAGN_W1_NFD_off.tolist()
    CLAGN_W1_zscore_mean_off[CLAGN_invert_indices_W1_off] *= -1
    CLAGN_W1_zscore_mean_off = CLAGN_W1_zscore_mean_off.tolist()

    CLAGN_W2_min_mjd_off = np.array(CLAGN_W2_min_mjd_off)
    CLAGN_W2_max_mjd_off = np.array(CLAGN_W2_max_mjd_off)
    CLAGN_W2_NFD_off = np.array(CLAGN_W2_NFD_off)
    CLAGN_W2_zscore_mean_off = np.array(CLAGN_W2_zscore_mean_off)
    # Find indices where min_mjd > max_mjd
    CLAGN_invert_indices_W2_off = CLAGN_W2_min_mjd_off > CLAGN_W2_max_mjd_off
    # Make corresponding W2_NFD values negative
    CLAGN_W2_NFD_off[CLAGN_invert_indices_W2_off] *= -1
    CLAGN_W2_NFD_off = CLAGN_W2_NFD_off.tolist()
    CLAGN_W2_zscore_mean_off[CLAGN_invert_indices_W2_off] *= -1
    CLAGN_W2_zscore_mean_off = CLAGN_W2_zscore_mean_off.tolist()

    CLAGN_mean_directional_NFD_off = [np.nanmean([x,y]) for x, y in zip(CLAGN_W1_NFD_off, CLAGN_W2_NFD_off)]
    CLAGN_mean_directional_zscore_off = [np.nanmean([x,y]) for x, y in zip(CLAGN_W1_zscore_mean_off, CLAGN_W2_zscore_mean_off)]


    CLAGN_W1_min_mjd_on = np.array(CLAGN_W1_min_mjd_on)
    CLAGN_W1_max_mjd_on = np.array(CLAGN_W1_max_mjd_on)
    CLAGN_W1_NFD_on = np.array(CLAGN_W1_NFD_on)
    CLAGN_W1_zscore_mean_on = np.array(CLAGN_W1_zscore_mean_on)
    # Find indices where min_mjd > max_mjd
    CLAGN_invert_indices_W1_on = CLAGN_W1_min_mjd_on > CLAGN_W1_max_mjd_on
    # Make corresponding W1_NFD values negative
    CLAGN_W1_NFD_on[CLAGN_invert_indices_W1_on] *= -1
    CLAGN_W1_NFD_on = CLAGN_W1_NFD_on.tolist()
    CLAGN_W1_zscore_mean_on[CLAGN_invert_indices_W1_on] *= -1
    CLAGN_W1_zscore_mean_on = CLAGN_W1_zscore_mean_on.tolist()

    CLAGN_W2_min_mjd_on = np.array(CLAGN_W2_min_mjd_on)
    CLAGN_W2_max_mjd_on = np.array(CLAGN_W2_max_mjd_on)
    CLAGN_W2_NFD_on = np.array(CLAGN_W2_NFD_on)
    CLAGN_W2_zscore_mean_on = np.array(CLAGN_W2_zscore_mean_on)
    # Find indices where min_mjd > max_mjd
    CLAGN_invert_indices_W2_on = CLAGN_W2_min_mjd_on > CLAGN_W2_max_mjd_on
    # Make corresponding W2_NFD values negative
    CLAGN_W2_NFD_on[CLAGN_invert_indices_W2_on] *= -1
    CLAGN_W2_NFD_on = CLAGN_W2_NFD_on.tolist()
    CLAGN_W2_zscore_mean_on[CLAGN_invert_indices_W2_on] *= -1
    CLAGN_W2_zscore_mean_on = CLAGN_W2_zscore_mean_on.tolist()

    CLAGN_mean_directional_NFD_on = [np.nanmean([x,y]) for x, y in zip(CLAGN_W1_NFD_on, CLAGN_W2_NFD_on)]
    CLAGN_mean_directional_zscore_on = [np.nanmean([x,y]) for x, y in zip(CLAGN_W1_zscore_mean_on, CLAGN_W2_zscore_mean_on)]

    max_NFD = np.nanmax(CLAGN_mean_directional_NFD_off+CLAGN_mean_directional_NFD_on)
    min_NFD = np.nanmin(CLAGN_mean_directional_NFD_off+CLAGN_mean_directional_NFD_on)
    max_zscore = np.nanmax(CLAGN_mean_directional_zscore_off+CLAGN_mean_directional_zscore_on)
    min_zscore = np.nanmin(CLAGN_mean_directional_zscore_off+CLAGN_mean_directional_zscore_on)

    CLAGN_median_NFD_off = np.nanmedian(CLAGN_mean_directional_NFD_off)
    CLAGN_median_zscore_off = np.nanmedian(CLAGN_mean_directional_zscore_off)
    CLAGN_median_NFD_on = np.nanmedian(CLAGN_mean_directional_NFD_on)
    CLAGN_median_zscore_on = np.nanmedian(CLAGN_mean_directional_zscore_on)

    plt.figure(figsize=(12, 7))
    plt.scatter(CLAGN_mean_directional_zscore_off, CLAGN_mean_directional_NFD_off, s=100, color='brown',  label=u'Turn-off CLAGN')
    plt.scatter(CLAGN_mean_directional_zscore_on, CLAGN_mean_directional_NFD_on, s=100, color='salmon',  label=u'Turn-on CLAGN')
    plt.axvline(0, color='black', linestyle=':')
    plt.axhline(0, color='black', linestyle=':')
    plt.xlim(1.1*min_zscore, 1.1*max_zscore)
    plt.ylim(1.1*min_NFD, 1.1*max_NFD)
    plt.tick_params(axis='both', labelsize=24, length=8, width=2)
    plt.xlabel("Directional Z-Score", fontsize = 24)
    plt.ylabel("Directional NFD", fontsize = 24)
    plt.title("NFD vs Z-score (Turn-off & Turn-on CLAGN)", fontsize = 24)
    ax = plt.gca()
    plt.text(0.04, 0.80, f'Turn-off CLAGN Median NFD = {CLAGN_median_NFD_off:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.04, 0.68, f'Turn-off CLAGN Median Z-Score = {CLAGN_median_zscore_off:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.04, 0.74, f'Turn-on CLAGN Median NFD = {CLAGN_median_NFD_on:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.04, 0.62, f'Turn-on CLAGN Median Z-Score = {CLAGN_median_zscore_on:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.legend(loc = 'best', fontsize=22)
    plt.tight_layout()
    plt.show()


if z_score_hist_direction_on_vs_off == 1:
    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[np.where(CLAGN_quantifying_change_data.iloc[:, 27].notna(),  
        CLAGN_quantifying_change_data.iloc[:, 27] >= bright_dim_W1,  
        CLAGN_quantifying_change_data.iloc[:, 30] >= bright_dim_W2)]
    
    CLAGN_quantifying_change_data_off = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4[Guo_table4['transition'] == 'turn-off'].iloc[:, 0])]
    CLAGN_W1_zscore_max_off = CLAGN_quantifying_change_data_off.iloc[:, 1].tolist()
    CLAGN_W1_zscore_min_off = CLAGN_quantifying_change_data_off.iloc[:, 3].tolist()
    CLAGN_W1_zscore_mean_off = [
        np.nanmean([abs(zmax), abs(zmin)]) 
        if not (np.isnan(zmax) and np.isnan(zmin))  # Check if both are NaN
        else np.nan  # Assign NaN if both are NaN
        for zmax, zmin in zip(CLAGN_W1_zscore_max_off, CLAGN_W1_zscore_min_off)
    ]
    CLAGN_W2_zscore_max_off = CLAGN_quantifying_change_data_off.iloc[:, 9].tolist()
    CLAGN_W2_zscore_min_off = CLAGN_quantifying_change_data_off.iloc[:, 11].tolist()
    CLAGN_W2_zscore_mean_off = [
        np.nanmean([abs(zmax), abs(zmin)]) 
        if not (np.isnan(zmax) and np.isnan(zmin))  # Check if both are NaN
        else np.nan  # Assign NaN if both are NaN
        for zmax, zmin in zip(CLAGN_W2_zscore_max_off, CLAGN_W2_zscore_min_off)
    ]
    CLAGN_W1_min_mjd_off = CLAGN_quantifying_change_data_off.iloc[:, 37].tolist()
    CLAGN_W1_max_mjd_off = CLAGN_quantifying_change_data_off.iloc[:, 38].tolist()
    CLAGN_W2_min_mjd_off = CLAGN_quantifying_change_data_off.iloc[:, 39].tolist()
    CLAGN_W2_max_mjd_off = CLAGN_quantifying_change_data_off.iloc[:, 40].tolist()
    
    CLAGN_quantifying_change_data_on = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4[Guo_table4['transition'] == 'turn-on'].iloc[:, 0])]
    CLAGN_W1_zscore_max_on = CLAGN_quantifying_change_data_on.iloc[:, 1].tolist()
    CLAGN_W1_zscore_min_on = CLAGN_quantifying_change_data_on.iloc[:, 3].tolist()
    CLAGN_W1_zscore_mean_on = [
        np.nanmean([abs(zmax), abs(zmin)]) 
        if not (np.isnan(zmax) and np.isnan(zmin))  # Check if both are NaN
        else np.nan  # Assign NaN if both are NaN
        for zmax, zmin in zip(CLAGN_W1_zscore_max_on, CLAGN_W1_zscore_min_on)
    ]
    CLAGN_W2_zscore_max_on = CLAGN_quantifying_change_data_on.iloc[:, 9].tolist()
    CLAGN_W2_zscore_min_on = CLAGN_quantifying_change_data_on.iloc[:, 11].tolist()
    CLAGN_W2_zscore_mean_on = [
        np.nanmean([abs(zmax), abs(zmin)]) 
        if not (np.isnan(zmax) and np.isnan(zmin))  # Check if both are NaN
        else np.nan  # Assign NaN if both are NaN
        for zmax, zmin in zip(CLAGN_W2_zscore_max_on, CLAGN_W2_zscore_min_on)
    ]
    CLAGN_W1_min_mjd_on = CLAGN_quantifying_change_data_on.iloc[:, 37].tolist()
    CLAGN_W1_max_mjd_on = CLAGN_quantifying_change_data_on.iloc[:, 38].tolist()
    CLAGN_W2_min_mjd_on = CLAGN_quantifying_change_data_on.iloc[:, 39].tolist()
    CLAGN_W2_max_mjd_on = CLAGN_quantifying_change_data_on.iloc[:, 40].tolist()


    CLAGN_W1_min_mjd_off = np.array(CLAGN_W1_min_mjd_off)
    CLAGN_W1_max_mjd_off = np.array(CLAGN_W1_max_mjd_off)
    CLAGN_W1_zscore_mean_off = np.array(CLAGN_W1_zscore_mean_off)
    # Find indices where min_mjd > max_mjd
    CLAGN_invert_indices_W1_off = CLAGN_W1_min_mjd_off > CLAGN_W1_max_mjd_off
    # Make corresponding W1_NFD values negative
    CLAGN_W1_zscore_mean_off[CLAGN_invert_indices_W1_off] *= -1
    CLAGN_W1_zscore_mean_off = CLAGN_W1_zscore_mean_off.tolist()

    CLAGN_W2_min_mjd_off = np.array(CLAGN_W2_min_mjd_off)
    CLAGN_W2_max_mjd_off = np.array(CLAGN_W2_max_mjd_off)
    CLAGN_W2_zscore_mean_off = np.array(CLAGN_W2_zscore_mean_off)
    # Find indices where min_mjd > max_mjd
    CLAGN_invert_indices_W2_off = CLAGN_W2_min_mjd_off > CLAGN_W2_max_mjd_off
    # Make corresponding W2_NFD values negative
    CLAGN_W2_zscore_mean_off[CLAGN_invert_indices_W2_off] *= -1
    CLAGN_W2_zscore_mean_off = CLAGN_W2_zscore_mean_off.tolist()

    CLAGN_mean_directional_zscore_off = [np.nanmean([x,y]) for x, y in zip(CLAGN_W1_zscore_mean_off, CLAGN_W2_zscore_mean_off)]


    CLAGN_W1_min_mjd_on = np.array(CLAGN_W1_min_mjd_on)
    CLAGN_W1_max_mjd_on = np.array(CLAGN_W1_max_mjd_on)
    CLAGN_W1_zscore_mean_on = np.array(CLAGN_W1_zscore_mean_on)
    # Find indices where min_mjd > max_mjd
    CLAGN_invert_indices_W1_on = CLAGN_W1_min_mjd_on > CLAGN_W1_max_mjd_on
    # Make corresponding W1_NFD values negative
    CLAGN_W1_zscore_mean_on[CLAGN_invert_indices_W1_on] *= -1
    CLAGN_W1_zscore_mean_on = CLAGN_W1_zscore_mean_on.tolist()

    CLAGN_W2_min_mjd_on = np.array(CLAGN_W2_min_mjd_on)
    CLAGN_W2_max_mjd_on = np.array(CLAGN_W2_max_mjd_on)
    CLAGN_W2_zscore_mean_on = np.array(CLAGN_W2_zscore_mean_on)
    # Find indices where min_mjd > max_mjd
    CLAGN_invert_indices_W2_on = CLAGN_W2_min_mjd_on > CLAGN_W2_max_mjd_on
    # Make corresponding W2_NFD values negative
    CLAGN_W2_zscore_mean_on[CLAGN_invert_indices_W2_on] *= -1
    CLAGN_W2_zscore_mean_on = CLAGN_W2_zscore_mean_on.tolist()

    CLAGN_mean_directional_zscore_on = [np.nanmean([x,y]) for x, y in zip(CLAGN_W1_zscore_mean_on, CLAGN_W2_zscore_mean_on)]

    CLAGN_median_zscore_off = np.nanmedian(CLAGN_mean_directional_zscore_off)
    CLAGN_median_zscore_on = np.nanmedian(CLAGN_mean_directional_zscore_on)


    zscore_off_binsize = (max(CLAGN_mean_directional_zscore_off) - min(CLAGN_mean_directional_zscore_off))/25
    zscore_on_binsize = (max(CLAGN_mean_directional_zscore_on) - min(CLAGN_mean_directional_zscore_on))/10
    bins_off = np.arange(min(CLAGN_mean_directional_zscore_off), max(CLAGN_mean_directional_zscore_off) + zscore_off_binsize, zscore_off_binsize)
    bins_on = np.arange(min(CLAGN_mean_directional_zscore_on), max(CLAGN_mean_directional_zscore_on) + zscore_on_binsize, zscore_on_binsize)

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False, gridspec_kw={'hspace': 0.25})

    # Histogram for CLAGN_mean_directional_zscore_off (inverted x-axis)
    ax1.hist(CLAGN_mean_directional_zscore_off, bins=bins_off, color='red', edgecolor='black', label='Turn-off CLAGN')
    ax1.axvline(CLAGN_median_zscore_off, color='black', linestyle='-', label=f'Turn-off CLAGN Median Z-score = {CLAGN_median_zscore_off:.2f}')
    ax1.set_ylabel("Frequency", fontsize=18)
    ax1.invert_xaxis()  # Invert x-axis
    ax1.tick_params(axis='both', labelsize=18, length=8, width=2)
    ax1.legend(loc='best', fontsize=18)
    ax1.set_title('Z-Score Distribution - Turn-Off vs Turn-On CLAGN', fontsize=20)

    # Histogram for CLAGN_mean_directional_zscore_on
    ax2.hist(CLAGN_mean_directional_zscore_on, bins=bins_on, color='darkred', edgecolor='black', label='Turn-on CLAGN')
    ax2.axvline(CLAGN_median_zscore_on, color='black', linestyle='-', label=f'Turn-on CLAGN Median Z-score = {CLAGN_median_zscore_on:.2f}')
    ax2.set_xlabel("Directional Z-Score", fontsize=18)
    ax2.set_ylabel("Frequency", fontsize=18)
    ax2.set_xlim(-ax1.get_xlim()[0], -ax1.get_xlim()[1])
    ax2.tick_params(axis='both', labelsize=18, length=8, width=2)
    ax2.legend(loc='best', fontsize=18)

    # plt.tight_layout()
    plt.show()


if main_MIR_line_split == 1:
    CLAGN_quantifying_change_data_Hbeta = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'Hbeta'].iloc[:, 0])]
    CLAGN_zscores_Hbeta = CLAGN_quantifying_change_data_Hbeta.iloc[:, 17].tolist()
    CLAGN_NFD_Hbeta = CLAGN_quantifying_change_data_Hbeta.iloc[:, 19].tolist()
    CLAGN_quantifying_change_data_Mg2 = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'Mg ii'].iloc[:, 0])]
    CLAGN_zscores_Mg2 = CLAGN_quantifying_change_data_Mg2.iloc[:, 17].tolist()
    CLAGN_NFD_Mg2 = CLAGN_quantifying_change_data_Mg2.iloc[:, 19].tolist()
    CLAGN_quantifying_change_data_C3_ = CLAGN_quantifying_change_data[CLAGN_quantifying_change_data.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'C iii]'].iloc[:, 0])]
    CLAGN_zscores_C3_ = CLAGN_quantifying_change_data_C3_.iloc[:, 17].tolist()
    CLAGN_NFD_C3_ = CLAGN_quantifying_change_data_C3_.iloc[:, 19].tolist()

    a = 0
    for zscore in CLAGN_zscores_Hbeta:
        if zscore > three_sigma_zscore:
            a += 1
    
    b = 0
    for zscore in CLAGN_zscores_Mg2:
        if zscore > three_sigma_zscore:
            b += 1

    c = 0
    for zscore in CLAGN_zscores_C3_:
        if zscore > three_sigma_zscore:
            c += 1

    d = 0
    for NFD in CLAGN_NFD_Hbeta:
        if NFD > three_sigma_norm_flux_diff:
            d += 1

    e = 0
    for NFD in CLAGN_NFD_Mg2:
        if NFD > three_sigma_norm_flux_diff:
            e += 1

    f = 0
    for NFD in CLAGN_NFD_C3_:
        if NFD > three_sigma_norm_flux_diff:
            f += 1

    plt.figure(figsize=(12, 7))
    plt.scatter(CLAGN_zscores_Hbeta, CLAGN_NFD_Hbeta, s=100, color='brown',  label=u'H\u03B2 CLAGN')
    plt.scatter(CLAGN_zscores_Mg2, CLAGN_NFD_Mg2, s=100, color='red',  label=u'Mg ii CLAGN')
    plt.scatter(CLAGN_zscores_C3_, CLAGN_NFD_C3_, s=100, color='salmon',  label=u'C iii] CLAGN')
    plt.axhline(y=three_sigma_norm_flux_diff, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2)
    plt.xlim(0, 1.05*max(CLAGN_zscores_Hbeta+CLAGN_zscores_Mg2+CLAGN_zscores_C3_))
    plt.ylim(0, 1.05*max(CLAGN_NFD_Hbeta+CLAGN_NFD_Mg2+CLAGN_NFD_C3_))
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.xlabel("Z-Score", fontsize = 26)
    plt.ylabel("NFD", fontsize = 26)
    plt.title("Comparing MIR Variability for different CLAGN BELs", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    ax = plt.gca()
    plt.text(0.99, 0.51, f'{a/len(CLAGN_zscores_Hbeta)*100:.1f}% Hbeta CLAGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.99, 0.45, f'{b/len(CLAGN_zscores_Mg2)*100:.1f}% Mg ii CLAGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.99, 0.39, f'{c/len(CLAGN_zscores_C3_)*100:.1f}% C iii] CLAGN > Z-Score Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.99, 0.15, f'{d/len(CLAGN_NFD_Hbeta)*100:.1f}% Hbeta CLAGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.99, 0.09, f'{e/len(CLAGN_NFD_Mg2)*100:.1f}% Mg ii CLAGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.99, 0.03, f'{f/len(CLAGN_NFD_C3_)*100:.1f}% C iii] CLAGN > NFD Threshold', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
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
    AGN_quantifying_change_data = AGN_quantifying_change_data[np.where(AGN_quantifying_change_data.iloc[:, 27].notna(),  
        AGN_quantifying_change_data.iloc[:, 27] >= bright_dim_W1,  
        AGN_quantifying_change_data.iloc[:, 30] >= bright_dim_W2)]
    AGN_norm_flux_diff_bright = AGN_quantifying_change_data.iloc[:, 19].tolist()
    AGN_norm_flux_diff_unc_bright = AGN_quantifying_change_data.iloc[:, 20].tolist()

    AGN_quantifying_change_data = pd.read_csv(f'AGN_Quantifying_Change_just_MIR_max_uncs_Sample_{my_sample}.csv')
    AGN_quantifying_change_data = AGN_quantifying_change_data[np.where(AGN_quantifying_change_data.iloc[:, 27].notna(),  
        AGN_quantifying_change_data.iloc[:, 27] < bright_dim_W1,  
        AGN_quantifying_change_data.iloc[:, 30] < bright_dim_W2)]
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
            
    ad_result_AGN = anderson_ksamp([AGN_norm_flux_diff_bright, AGN_norm_flux_diff_dim])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.hist(AGN_norm_flux_diff_bright, bins=AGN_bins_flux_diff, color='black', histtype='step', linewidth=2, label='Bright Non-CL AGN')
    ax1.hist(AGN_norm_flux_diff_dim, bins=AGN_bins_flux_diff, color='gray', alpha=0.7, label='Dim Non-CL AGN')
    ax1.axvline(median_norm_flux_diff_AGN_bright, linewidth=2, linestyle='-', color='blue', label=f'Bright Non-CL AGN Median = {median_norm_flux_diff_AGN_bright:.2f}')
    ax1.axvline(median_norm_flux_diff_AGN_dim, linewidth=2, linestyle='--', color='blue', label=f'Dim Non-CL AGN Median = {median_norm_flux_diff_AGN_dim:.2f}')
    ax1.axvline(three_sigma_norm_flux_diff_bright, linewidth=2, linestyle=':', color='black', label=f'{a/len(AGN_norm_flux_diff_bright)*100:.1f}% Bright Non-CL AGN > Bright Threshold = {three_sigma_norm_flux_diff_bright:.2f}')
    ax1.axvline(three_sigma_norm_flux_diff_dim, linewidth=2, linestyle='-.', color='grey', label=f'{b/len(AGN_norm_flux_diff_dim)*100:.1f}% Dim Non-CL AGN > Dim Threshold = {three_sigma_norm_flux_diff_dim:.2f}')
    ax1.plot((x_start_threshold_bright, x_end_threshold_bright), (height_bright+0.75, height_bright+0.75), linewidth=2, color='sienna')
    ax1.plot((x_start_threshold_dim, x_end_threshold_dim), (height_dim+0.25, height_dim+0.25), linewidth=2, color='darkorange')
    ax1.text(1.42, height_bright+1.25, f'3X Median Bright Uncertainty = {3*median_norm_flux_diff_AGN_unc_bright:.2f}', 
            ha='right', va='center', fontsize=14, color='sienna')
    ax1.text(2.42, 5, f'3X Median Dim Uncertainty = {3*median_norm_flux_diff_AGN_unc_dim:.2f}', 
            ha='right', va='center', fontsize=14, color='darkorange')
    ax1.text(-0.1, 12, f'Non-CL AGN', ha='left', va='center', fontsize=18, color='blue')
    ax1.text(2.42, 2, f'AD test - non-CL AGN p-value < {ad_result_AGN.pvalue:.0e}', fontsize=13, ha='right', va='center')
    ax1.set_ylabel('Non-CL AGN Frequency', color='black', fontsize=18)
    ax1.tick_params(axis='both', labelsize=18, length=8, width=2)
    ax1.legend(loc='upper right')

    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[np.where(CLAGN_quantifying_change_data.iloc[:, 27].notna(),  
        CLAGN_quantifying_change_data.iloc[:, 27] >= bright_dim_W1,  
        CLAGN_quantifying_change_data.iloc[:, 30] >= bright_dim_W2)]
    CLAGN_norm_flux_diff_bright = CLAGN_quantifying_change_data.iloc[:, 19].tolist()
    
    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[np.where(CLAGN_quantifying_change_data.iloc[:, 27].notna(),  
        CLAGN_quantifying_change_data.iloc[:, 27] < bright_dim_W1,  
        CLAGN_quantifying_change_data.iloc[:, 30] < bright_dim_W2)]
    CLAGN_norm_flux_diff_dim = CLAGN_quantifying_change_data.iloc[:, 19].tolist()

    CLAGN_norm_flux_diff_all = CLAGN_norm_flux_diff_bright+CLAGN_norm_flux_diff_dim
    median_norm_flux_diff_CLAGN_bright = np.nanmedian(CLAGN_norm_flux_diff_bright)
    median_norm_flux_diff_CLAGN_dim = np.nanmedian(CLAGN_norm_flux_diff_dim)

    CLAGN_flux_diff_binsize = (max(CLAGN_norm_flux_diff_all) - min(CLAGN_norm_flux_diff_all))/20
    CLAGN_bins_flux_diff = np.arange(min(CLAGN_norm_flux_diff_all), max(CLAGN_norm_flux_diff_all) + CLAGN_flux_diff_binsize, CLAGN_flux_diff_binsize)
    
    c = 0
    for NFD in CLAGN_norm_flux_diff_bright:
        if NFD > three_sigma_norm_flux_diff_bright:
            c += 1
    
    d = 0
    for NFD in CLAGN_norm_flux_diff_dim:
        if NFD > three_sigma_norm_flux_diff_dim:
            d += 1

    # ks_statistic_bright, p_value_bright = ks_2samp(AGN_norm_flux_diff_bright, CLAGN_norm_flux_diff_bright)
    # ks_statistic_dim, p_value_dim = ks_2samp(AGN_norm_flux_diff_dim, CLAGN_norm_flux_diff_dim)
    # ad_result_bright = anderson_ksamp([AGN_norm_flux_diff_bright, CLAGN_norm_flux_diff_bright])
    # ad_result_dim = anderson_ksamp([AGN_norm_flux_diff_dim, CLAGN_norm_flux_diff_dim])

    ad_result_CLAGN = anderson_ksamp([CLAGN_norm_flux_diff_bright, CLAGN_norm_flux_diff_dim])

    ax2.hist(CLAGN_norm_flux_diff_bright, bins=CLAGN_bins_flux_diff, color='black', histtype='step', linewidth=2, label='Bright CLAGN')
    ax2.hist(CLAGN_norm_flux_diff_dim, bins=CLAGN_bins_flux_diff, color='gray', alpha=0.7, label='Dim CLAGN')
    ax2.axvline(median_norm_flux_diff_CLAGN_bright, linewidth=2, linestyle='-', color='red', label=f'Bright CLAGN Median = {median_norm_flux_diff_CLAGN_bright:.2f}')
    ax2.axvline(median_norm_flux_diff_CLAGN_dim, linewidth=2, linestyle='--', color='red', label=f'Dim CLAGN Median = {median_norm_flux_diff_CLAGN_dim:.2f}')
    ax2.axvline(three_sigma_norm_flux_diff_bright, linewidth=2, linestyle=':', color='black', label=f'{c/len(CLAGN_norm_flux_diff_bright)*100:.1f}% Bright CLAGN > Bright Threshold = {three_sigma_norm_flux_diff_bright:.2f}')
    ax2.axvline(three_sigma_norm_flux_diff_dim, linewidth=2, linestyle='-.', color='grey', label=f'{d/len(CLAGN_norm_flux_diff_dim)*100:.1f}% Dim CLAGN > Dim Threshold = {three_sigma_norm_flux_diff_dim:.2f}')
    # ax2.text(-0.1, 4.75, f'KS test - bright objects. P-value = {p_value_bright:.2f}', fontsize = 10, ha='left', va='center')
    # ax2.text(-0.1, 4.25, f'KS test - dim objects. P-value = {p_value_dim:.2f}', fontsize = 10, ha='left', va='center')
    # ax2.text(-0.1, 4.75, f'AD test - bright objects. P-value = {ad_result_bright.pvalue:.2f}', fontsize = 10, ha='left', va='center')
    # ax2.text(-0.1, 4.25, f'AD test - dim objects. P-value = {ad_result_dim.pvalue:.2f}', fontsize = 10, ha='left', va='center')
    ax2.text(-0.1, 5.25, f'AD test - CLAGN p-value = {ad_result_CLAGN.pvalue:.2f}', fontsize = 13, ha='left', va='center')
    ax2.text(-0.1, 5.75, f'CLAGN', ha='left', va='center', fontsize=18, color='red')
    ax2.set_xlabel('NFD', fontsize=18)
    ax2.set_ylabel('CLAGN Frequency', color='black', fontsize=18)
    ax2.tick_params(axis='both', labelsize=18, length=8, width=2)
    ax2.legend(loc='upper right')

    plt.suptitle(f'NFD Distribution - CLAGN & Non-CL AGN Sample {my_sample}', fontsize=20)
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
    AGN_quantifying_change_data = AGN_quantifying_change_data[np.where(AGN_quantifying_change_data.iloc[:, 27].notna(),  
        AGN_quantifying_change_data.iloc[:, 27] >= bright_dim_W1,  
        AGN_quantifying_change_data.iloc[:, 30] >= bright_dim_W2)]
    AGN_z_score_bright = AGN_quantifying_change_data.iloc[:, 17].tolist()
    AGN_z_score_unc_bright = AGN_quantifying_change_data.iloc[:, 18].tolist()

    AGN_quantifying_change_data = pd.read_csv(f'AGN_Quantifying_Change_just_MIR_max_uncs_Sample_{my_sample}.csv')
    AGN_quantifying_change_data = AGN_quantifying_change_data[np.where(AGN_quantifying_change_data.iloc[:, 27].notna(),  
        AGN_quantifying_change_data.iloc[:, 27] < bright_dim_W1,  
        AGN_quantifying_change_data.iloc[:, 30] < bright_dim_W2)]
    AGN_z_score_dim = AGN_quantifying_change_data.iloc[:, 17].tolist()
    AGN_z_score_unc_dim = AGN_quantifying_change_data.iloc[:, 18].tolist()

    AGN_z_score_all = AGN_z_score_bright+AGN_z_score_dim
    median_z_score_AGN_bright = np.nanmedian(AGN_z_score_bright)
    median_z_score_AGN_unc_bright = np.nanmedian(AGN_z_score_unc_bright)
    three_sigma_z_score_bright = median_z_score_AGN_bright + 3*median_z_score_AGN_unc_bright
    median_z_score_AGN_dim = np.nanmedian(AGN_z_score_dim)
    median_z_score_AGN_unc_dim = np.nanmedian(AGN_z_score_unc_dim)
    three_sigma_z_score_dim = median_z_score_AGN_dim + 3*median_z_score_AGN_unc_dim

    AGN_z_score_binsize = (max(AGN_z_score_all) - min(AGN_z_score_all))/25
    AGN_bins_z_score = np.arange(min(AGN_z_score_all), max(AGN_z_score_all) + AGN_z_score_binsize, AGN_z_score_binsize)
    x_start_threshold_bright = median_z_score_AGN_bright - 3*median_z_score_AGN_unc_bright
    x_end_threshold_bright = median_z_score_AGN_bright + 3*median_z_score_AGN_unc_bright
    counts_bright, bin_edges = np.histogram(AGN_z_score_bright, bins=AGN_bins_z_score)
    height_bright = max(counts_bright)/2

    x_start_threshold_dim = median_z_score_AGN_dim - 3*median_z_score_AGN_unc_dim
    x_end_threshold_dim = median_z_score_AGN_dim + 3*median_z_score_AGN_unc_dim
    counts_dim, bin_edges = np.histogram(AGN_z_score_dim, bins=AGN_bins_z_score)
    height_dim = max(counts_dim)/2

    a = 0
    for zs in AGN_z_score_bright:
        if zs > three_sigma_z_score_bright:
            a += 1
    
    b = 0
    for zs in AGN_z_score_dim:
        if zs > three_sigma_z_score_dim:
            b += 1
    
    ad_result_AGN = anderson_ksamp([AGN_z_score_bright, AGN_z_score_dim])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)  
    ax1.hist(AGN_z_score_bright, bins=AGN_bins_z_score,  color='black', histtype='step', linewidth=2, label='Bright Non-CL AGN')
    ax1.hist(AGN_z_score_dim, bins=AGN_bins_z_score, color='gray', alpha=0.7, label='Dim Non-CL AGN')
    ax1.axvline(median_z_score_AGN_bright, linewidth=2, linestyle='-', color='blue', label=f'Bright Non-CL AGN Median = {median_z_score_AGN_bright:.2f}')
    ax1.axvline(median_z_score_AGN_dim, linewidth=2, linestyle='--', color='blue', label=f'Dim Non-CL AGN Median = {median_z_score_AGN_dim:.2f}')
    ax1.axvline(three_sigma_z_score_bright, linewidth=2, linestyle=':', color='black', label=f'{a/len(AGN_z_score_bright)*100:.1f}% Bright Non-CL AGN > Bright Threshold = {three_sigma_z_score_bright:.2f}')
    ax1.axvline(three_sigma_z_score_dim, linewidth=2, linestyle='-', color='black', label=f'{b/len(AGN_z_score_dim)*100:.1f}% Dim Non-CL AGN > Dim Threshold = {three_sigma_z_score_dim:.2f}')
    ax1.plot((x_start_threshold_bright, x_end_threshold_bright), (height_bright+0.75, height_bright+0.75), linewidth=2, color='sienna')
    ax1.plot((x_start_threshold_dim, x_end_threshold_dim), (height_dim+0.25, height_dim+0.25), linewidth=2, color='darkorange')
    ax1.text(x_end_threshold_bright+1, height_bright+1.25, f'3X Median Bright Non-CL AGN Z-Score Uncertainty = {3*median_z_score_AGN_unc_bright:.2f}', 
            ha='left', va='center', fontsize=14, color='sienna')
    ax1.text(x_end_threshold_bright+1, height_bright+8.25, f'3X Median Dim Non-CL AGN Z-Score Uncertainty = {3*median_z_score_AGN_unc_dim:.2f}', 
            ha='left', va='center', fontsize=14, color='darkorange')
    ax1.text(60, 5, f'AD test - non-CL AGN p-value < {ad_result_AGN.pvalue:.0e}', fontsize=16, ha='right', va='center')
    ax1.text(20, 47, f'Non-CL AGN', ha='left', va='center', fontsize=20, color='blue')
    ax1.tick_params(axis='both', labelsize=18, length=8, width=2)
    ax1.set_ylabel('Non-CL AGN Frequency', color='black', fontsize=18)
    ax1.legend(loc='upper right')

    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[np.where(CLAGN_quantifying_change_data.iloc[:, 27].notna(),  
        CLAGN_quantifying_change_data.iloc[:, 27] >= bright_dim_W1,  
        CLAGN_quantifying_change_data.iloc[:, 30] >= bright_dim_W2)]
    CLAGN_z_score_bright = CLAGN_quantifying_change_data.iloc[:, 17].tolist()

    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[np.where(CLAGN_quantifying_change_data.iloc[:, 27].notna(),  
        CLAGN_quantifying_change_data.iloc[:, 27] < bright_dim_W1,  
        CLAGN_quantifying_change_data.iloc[:, 30] < bright_dim_W2)]
    CLAGN_z_score_dim = CLAGN_quantifying_change_data.iloc[:, 17].tolist()

    CLAGN_z_score_all = CLAGN_z_score_bright+CLAGN_z_score_dim
    median_z_score_CLAGN_bright = np.nanmedian(CLAGN_z_score_bright)
    median_z_score_CLAGN_dim = np.nanmedian(CLAGN_z_score_dim)

    CLAGN_z_score_binsize = (max(CLAGN_z_score_all) - min(CLAGN_z_score_all))/25
    CLAGN_bins_z_score = np.arange(min(CLAGN_z_score_all), max(CLAGN_z_score_all) + 2*CLAGN_z_score_binsize, CLAGN_z_score_binsize)
    
    c = 0
    for zs in CLAGN_z_score_bright:
        if zs > three_sigma_z_score_bright:
            c += 1
    
    d = 0
    for zs in CLAGN_z_score_dim:
        if zs > three_sigma_z_score_dim:
            d += 1

    # ks_statistic_bright, p_value_bright = ks_2samp(AGN_z_score_bright, CLAGN_z_score_bright)
    # ks_statistic_dim, p_value_dim = ks_2samp(AGN_z_score_dim, CLAGN_z_score_dim)
    # ad_result_bright = anderson_ksamp([AGN_z_score_bright, CLAGN_z_score_bright])
    # ad_result_dim = anderson_ksamp([AGN_z_score_dim, CLAGN_z_score_dim])
    # print(f'AD test p-value, dim distributions = {ad_result_dim.pvalue}')

    ad_result_CLAGN = anderson_ksamp([CLAGN_z_score_bright, CLAGN_z_score_dim])

    max_line = max([median_z_score_CLAGN_bright, median_z_score_CLAGN_dim,
                    three_sigma_z_score_bright, three_sigma_z_score_dim])

    ax2.hist(CLAGN_z_score_bright, bins=CLAGN_bins_z_score,  color='black', histtype='step', linewidth=2, label='Bright CLAGN')
    ax2.hist(CLAGN_z_score_dim, bins=CLAGN_bins_z_score, color='gray', alpha=0.7, label='Dim CLAGN')
    ax2.axvline(median_z_score_CLAGN_bright, linewidth=2, linestyle='-', color='red', label=f'Bright CLAGN Median = {median_z_score_CLAGN_bright:.2f}')
    ax2.axvline(median_z_score_CLAGN_dim, linewidth=2, linestyle='--', color='red', label=f'Dim CLAGN Median = {median_z_score_CLAGN_dim:.2f}')
    ax2.axvline(three_sigma_z_score_bright, linewidth=2, linestyle=':', color='black', label=f'{c/len(CLAGN_z_score_bright)*100:.1f}% Bright CLAGN > Bright Threshold = {three_sigma_z_score_bright:.2f}')
    ax2.axvline(three_sigma_z_score_dim, linewidth=2, linestyle='-', color='black', label=f'{d/len(CLAGN_z_score_dim)*100:.1f}% Dim CLAGN > Dim Threshold = {three_sigma_z_score_dim:.2f}')
    # ax2.text(max_line+1, 10, f'KS test - bright objects. P-value = {p_value_bright:.2f}', fontsize = 10, ha='left', va='center')
    # ax2.text(max_line+1, 8, f'KS test - dim objects. P-value = {p_value_dim:.2f}', fontsize = 10, ha='left', va='center')
    # ax2.text(max_line+1, 10, f'AD test - bright objects. P-value = {ad_result_bright.pvalue:.2f}', fontsize = 10, ha='left', va='center')
    # ax2.text(max_line+1, 8, f'AD test - dim objects. P-value = {ad_result_dim.pvalue:.2f}', fontsize = 10, ha='left', va='center')
    ax2.text(60, 3, f'AD test - CLAGN p-value < {ad_result_CLAGN.pvalue:.0e}', fontsize = 16, ha='right', va='center')
    ax2.text(20, 13, f'CLAGN', ha='left', va='center', fontsize=20, color='red')
    ax2.set_xlabel('Z-Score', fontsize=18)
    ax2.set_ylabel('CLAGN Frequency', color='black', fontsize=18)
    ax2.tick_params(axis='both', labelsize=18, length=8, width=2)
    ax2.legend(loc='upper right')

    plt.suptitle(f'Z-Score Distribution - CLAGN & Non-CL AGN Sample {my_sample}', fontsize=20)
    plt.tight_layout()
    plt.show()


# # #Creating a 2d plot for UV normalised flux difference & z score:
if UV_MIRZ == 1:
    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_zscores, AGN_norm_flux_diff_UV, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_zscores, CLAGN_norm_flux_diff_UV, s=100, color='red',  label='CLAGN')
    # plt.errorbar(AGN_zscores, AGN_norm_flux_diff, xerr=AGN_zscore_uncs, yerr=AGN_norm_flux_diff_UV_unc, fmt='o', color='blue', label='Non-CL AGN')
    # plt.errorbar(CLAGN_zscores, CLAGN_norm_flux_diff, xerr=CLAGN_zscore_uncs, yerr=CLAGN_norm_flux_diff_UV_unc, fmt='o', color='red',  label='CLAGN')
    plt.axhline(y=three_sigma_UV_NFD, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2)
    plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
    plt.ylim(1.05*(min(CLAGN_norm_flux_diff_UV+AGN_norm_flux_diff_UV)), 1.05*max(CLAGN_norm_flux_diff_UV+AGN_norm_flux_diff_UV))
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xlabel("Interpolated Z-Score", fontsize = 26)
    plt.ylabel("UV NFD", fontsize = 26)
    plt.title("UV NFD vs Interpolated Z-Score", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
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
    plt.scatter(CLAGN_norm_flux_diff, CLAGN_norm_flux_diff_UV, s=100, color='red',  label='CLAGN')
    # plt.errorbar(AGN_zscores, AGN_norm_flux_diff, xerr=AGN_zscore_uncs, yerr=AGN_norm_flux_diff_UV_unc, fmt='o', color='blue', label='Non-CL AGN')
    # plt.errorbar(CLAGN_zscores, CLAGN_norm_flux_diff, xerr=CLAGN_zscore_uncs, yerr=CLAGN_norm_flux_diff_UV_unc, fmt='o', color='red',  label='CLAGN')
    plt.axhline(y=three_sigma_UV_NFD, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.axvline(x=three_sigma_norm_flux_diff, color='black', linestyle='--', linewidth=2)
    plt.xlim(0, 1.05*max(AGN_norm_flux_diff+CLAGN_norm_flux_diff))
    plt.ylim(1.05*(min(CLAGN_norm_flux_diff_UV+AGN_norm_flux_diff_UV)), 1.05*max(CLAGN_norm_flux_diff_UV+AGN_norm_flux_diff_UV))
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xlabel("Interpolated NFD", fontsize = 26)
    plt.ylabel("UV NFD", fontsize = 26)
    plt.title("UV NFD vs Interpolated MIR NFD", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
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
    plt.scatter(CLAGN_zscores_MIR, CLAGN_zscores_UV, s=100, color='red',  label='CLAGN')
    plt.plot(x, x, color='black', linestyle='-', label = 'y=x') #add a y=x line
    plt.xlim(0, 1.05*max(CLAGN_zscores_MIR+AGN_zscores_MIR))
    plt.ylim(0, 1.05*max(CLAGN_zscores_UV+AGN_zscores_UV))
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.xlabel("Max/Min Z-Score", fontsize = 26)
    plt.ylabel("Interpolated Z-Score", fontsize = 26)
    plt.title(f"Comparing MIR Z-Score & UV Z-Score (Sample {my_sample})", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if UVNFD_MIRZ == 1:
    AGN_UV_all = pd.read_csv(f'AGN_Quantifying_Change_Sample_{my_sample}_UV_all.csv')
    AGN_UV_all = AGN_UV_all.dropna(subset=[AGN_UV_all.columns[1]])
    AGN_UV_names = AGN_UV_all.iloc[:, 0]
    AGN_UV_NFD = AGN_UV_all.iloc[:, 1].tolist()
    AGN_UV_NFD_unc = AGN_UV_all.iloc[:, 2].tolist()

    AGN_MIR_all = pd.read_csv(f'AGN_Quantifying_Change_just_MIR_max_uncs_Sample_{my_sample}.csv')
    AGN_MIR_all = AGN_MIR_all[np.where(AGN_MIR_all.iloc[:, 27].notna(),  
        AGN_MIR_all.iloc[:, 27] >= bright_dim_W1,  
        AGN_MIR_all.iloc[:, 30] >= bright_dim_W2)]
    AGN_MIR_names = AGN_MIR_all.iloc[:, 0]
    AGN_zscores_MIR = AGN_MIR_all.iloc[:, 17]
    AGN_zscores_MIR_unc = AGN_MIR_all.iloc[:, 18]

    AGN_name_to_UV_NFD = dict(zip(AGN_UV_names, zip(AGN_UV_NFD, AGN_UV_NFD_unc)))
    AGN_name_to_zs_MIR = dict(zip(AGN_MIR_names, zip(AGN_zscores_MIR, AGN_zscores_MIR_unc)))
    all_names = set(AGN_UV_names).union(set(AGN_MIR_names))
    matched_names = set(AGN_name_to_UV_NFD.keys()).intersection(set(AGN_name_to_zs_MIR.keys()))

    AGN_UV_NFD, AGN_UV_NFD_unc = zip(*[AGN_name_to_UV_NFD[name] for name in matched_names])
    AGN_zscores_MIR, AGN_zscores_MIR_unc = zip(*[AGN_name_to_zs_MIR[name] for name in matched_names])

    median_AGN_UV_NFD = np.nanmedian(AGN_UV_NFD)
    median_AGN_UV_NFD_unc = np.nanmedian(AGN_UV_NFD_unc)
    three_sigma_UV_NFD = median_AGN_UV_NFD + 3*median_AGN_UV_NFD_unc

    median_AGN_zscores_MIR = np.nanmedian(AGN_zscores_MIR)
    median_AGN_zscores_MIR_unc = np.nanmedian(AGN_zscores_MIR_unc)
    three_sigma_zscores_MIR = median_AGN_zscores_MIR + 3*median_AGN_zscores_MIR_unc

    CLAGN_UV_all = pd.read_csv('CLAGN_Quantifying_Change_UV_all.csv')
    CLAGN_UV_all = CLAGN_UV_all.dropna(subset=[CLAGN_UV_all.columns[1]])
    CLAGN_UV_names = CLAGN_UV_all.iloc[:, 0]
    CLAGN_UV_NFD = CLAGN_UV_all.iloc[:, 1].tolist()

    CLAGN_MIR_all = pd.read_csv(f'CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_MIR_all = CLAGN_MIR_all[np.where(CLAGN_MIR_all.iloc[:, 27].notna(),  
        CLAGN_MIR_all.iloc[:, 27] >= bright_dim_W1,  
        CLAGN_MIR_all.iloc[:, 30] >= bright_dim_W2)]
    CLAGN_MIR_names = CLAGN_MIR_all.iloc[:, 0]
    CLAGN_zscores_MIR = CLAGN_MIR_all.iloc[:, 17]

    CLAGN_name_to_UV_NFD = dict(zip(CLAGN_UV_names, CLAGN_UV_NFD))
    CLAGN_name_to_zs_MIR = dict(zip(CLAGN_MIR_names, CLAGN_zscores_MIR))
    all_names = set(CLAGN_UV_names).union(set(CLAGN_MIR_names))

    matched_names = set(CLAGN_name_to_UV_NFD.keys()).intersection(set(CLAGN_name_to_zs_MIR.keys()))
    CLAGN_UV_NFD = [CLAGN_name_to_UV_NFD[name] for name in matched_names]
    CLAGN_zscores_MIR = [CLAGN_name_to_zs_MIR[name] for name in matched_names]

    a = 0
    for my_NFD in AGN_UV_NFD:
        if my_NFD > three_sigma_UV_NFD:
            a += 1
    
    b = 0
    for my_zs in AGN_zscores_MIR:
        if my_zs > three_sigma_zscores_MIR:
            b += 1
    
    c = 0
    for my_NFD in CLAGN_UV_NFD:
        if my_NFD > three_sigma_UV_NFD:
            c += 1
    
    d = 0
    for my_zs in CLAGN_zscores_MIR:
        if my_zs > three_sigma_zscores_MIR:
            d += 1

    print(f'Number of CLAGN analysed = {len(CLAGN_UV_NFD)}')
    print(f'Number of AGN analysed = {len(AGN_UV_NFD)}')


    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_zscores_MIR, AGN_UV_NFD, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_zscores_MIR, CLAGN_UV_NFD, s=100, color='red',  label='CLAGN')
    plt.axvline(x=three_sigma_zscores_MIR, color='black', linestyle='--', linewidth=2, label = 'Threshold')
    plt.axhline(y=three_sigma_UV_NFD, color='black', linestyle='--', linewidth=2)
    # plt.xlim(0, 1.05*max(CLAGN_zscores_MIR+AGN_zscores_MIR))
    # plt.ylim(0, 1.05*max(CLAGN_UV_NFD+AGN_UV_NFD))
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.xlabel("Z-Score", fontsize = 26) #max/min z-score
    plt.ylabel("UV NFD", fontsize = 26)
    plt.title(f"Comparing UV NFD and MIR Z-Score - Bright Objects", fontsize = 28)
    ax = plt.gca()
    plt.text(0.32, 0.92, f'{c/len(CLAGN_UV_NFD)*100:.0f}% CLAGN,', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.32, 0.86, f'{a/len(AGN_UV_NFD)*100:.0f}% Non-CL AGN', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.32, 0.80, f'> UV NFD Threshold', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.65, 0.52, f'{d/len(CLAGN_zscores_MIR)*100:.0f}% CLAGN,', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.65, 0.46, f'{b/len(AGN_zscores_MIR)*100:.0f}% Non-CL AGN', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.65, 0.40, f'> Z-Score Threshold', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.legend(loc = 'best', fontsize=25)
    plt.tight_layout()
    plt.show()


if UVNFD_MIRNFD == 1:
    AGN_UV_all = pd.read_csv(f'AGN_Quantifying_Change_Sample_{my_sample}_UV_all.csv')
    AGN_UV_all = AGN_UV_all.dropna(subset=[AGN_UV_all.columns[1]])
    AGN_UV_names = AGN_UV_all.iloc[:, 0]
    AGN_UV_NFD = AGN_UV_all.iloc[:, 1].tolist()
    AGN_UV_NFD_unc = AGN_UV_all.iloc[:, 2].tolist()

    AGN_MIR_all = pd.read_csv(f'AGN_Quantifying_Change_just_MIR_max_uncs_Sample_{my_sample}.csv')
    AGN_MIR_all = AGN_MIR_all[np.where(AGN_MIR_all.iloc[:, 27].notna(),  
        AGN_MIR_all.iloc[:, 27] >= bright_dim_W1,  
        AGN_MIR_all.iloc[:, 30] >= bright_dim_W2)]
    AGN_MIR_names = AGN_MIR_all.iloc[:, 0]
    AGN_NFD_MIR = AGN_MIR_all.iloc[:, 19]
    AGN_NFD_MIR_unc = AGN_MIR_all.iloc[:, 20]

    AGN_name_to_UV_NFD = dict(zip(AGN_UV_names, zip(AGN_UV_NFD, AGN_UV_NFD_unc)))
    AGN_name_to_NFD_MIR = dict(zip(AGN_MIR_names, zip(AGN_NFD_MIR, AGN_NFD_MIR_unc)))
    all_names = set(AGN_UV_names).union(set(AGN_MIR_names))
    matched_names = set(AGN_name_to_UV_NFD.keys()).intersection(set(AGN_name_to_NFD_MIR.keys()))

    AGN_UV_NFD, AGN_UV_NFD_unc = zip(*[AGN_name_to_UV_NFD[name] for name in matched_names])
    AGN_NFD_MIR, AGN_NFD_MIR_unc = zip(*[AGN_name_to_NFD_MIR[name] for name in matched_names])

    median_AGN_UV_NFD = np.nanmedian(AGN_UV_NFD)
    median_AGN_UV_NFD_unc = np.nanmedian(AGN_UV_NFD_unc)
    three_sigma_UV_NFD = median_AGN_UV_NFD + 3*median_AGN_UV_NFD_unc

    median_AGN_NFD_MIR = np.nanmedian(AGN_NFD_MIR)
    median_AGN_NFD_MIR_unc = np.nanmedian(AGN_NFD_MIR_unc)
    three_sigma_NFD_MIR = median_AGN_NFD_MIR + 3*median_AGN_NFD_MIR_unc

    CLAGN_UV_all = pd.read_csv('CLAGN_Quantifying_Change_UV_all.csv')
    CLAGN_UV_all = CLAGN_UV_all.dropna(subset=[CLAGN_UV_all.columns[1]])
    CLAGN_UV_names = CLAGN_UV_all.iloc[:, 0]
    CLAGN_UV_NFD = CLAGN_UV_all.iloc[:, 1].tolist()

    CLAGN_MIR_all = pd.read_csv(f'CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_MIR_all = CLAGN_MIR_all[np.where(CLAGN_MIR_all.iloc[:, 27].notna(),  
        CLAGN_MIR_all.iloc[:, 27] >= bright_dim_W1,  
        CLAGN_MIR_all.iloc[:, 30] >= bright_dim_W2)]
    CLAGN_MIR_names = CLAGN_MIR_all.iloc[:, 0]
    CLAGN_NFD_MIR = CLAGN_MIR_all.iloc[:, 19]

    CLAGN_name_to_UV_NFD = dict(zip(CLAGN_UV_names, CLAGN_UV_NFD))
    CLAGN_name_to_NFD_MIR = dict(zip(CLAGN_MIR_names, CLAGN_NFD_MIR))
    all_names = set(CLAGN_UV_names).union(set(CLAGN_MIR_names))

    matched_names = set(CLAGN_name_to_UV_NFD.keys()).intersection(set(CLAGN_name_to_NFD_MIR.keys()))
    CLAGN_UV_NFD = [CLAGN_name_to_UV_NFD[name] for name in matched_names]
    CLAGN_NFD_MIR = [CLAGN_name_to_NFD_MIR[name] for name in matched_names]

    a = 0
    for my_NFD in AGN_UV_NFD:
        if my_NFD > three_sigma_UV_NFD:
            a += 1
    
    b = 0
    for my_zs in AGN_NFD_MIR:
        if my_zs > three_sigma_NFD_MIR:
            b += 1
    
    c = 0
    for my_NFD in CLAGN_UV_NFD:
        if my_NFD > three_sigma_UV_NFD:
            c += 1
    
    d = 0
    for my_zs in CLAGN_NFD_MIR:
        if my_zs > three_sigma_NFD_MIR:
            d += 1

    print(f'Number of CLAGN analysed = {len(CLAGN_UV_NFD)}')
    print(f'Number of AGN analysed = {len(AGN_UV_NFD)}')


    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_NFD_MIR, AGN_UV_NFD, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_NFD_MIR, CLAGN_UV_NFD, s=100, color='red',  label='CLAGN')
    plt.axvline(x=three_sigma_NFD_MIR, color='black', linestyle='--', linewidth=2, label = 'Threshold')
    plt.axhline(y=three_sigma_UV_NFD, color='black', linestyle='--', linewidth=2)
    # plt.xlim(0, 1.05*max(CLAGN_NFD_MIR+AGN_NFD_MIR))
    # plt.ylim(0, 1.05*max(CLAGN_UV_NFD+AGN_UV_NFD))
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.xlabel("MIR NFD", fontsize = 26) #max/min NFD
    plt.ylabel("UV NFD", fontsize = 26)
    plt.title(f"Comparing UV NFD and MIR NFD", fontsize = 28)
    ax = plt.gca()
    plt.text(0.02, 0.88, f'{c/len(CLAGN_UV_NFD)*100:.0f}% CLAGN,', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.02, 0.82, f'{a/len(AGN_UV_NFD)*100:.0f}% Non-CL AGN', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.02, 0.76, f'> UV NFD Threshold', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.65, 0.52, f'{d/len(CLAGN_NFD_MIR)*100:.0f}% CLAGN,', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.65, 0.46, f'{b/len(AGN_NFD_MIR)*100:.0f}% Non-CL AGN', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.text(0.65, 0.40, f'> MIR NFD Threshold', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    plt.legend(loc = 'best', fontsize=25)
    plt.tight_layout()
    plt.show()


if UV_NFD_dist == 1:
    AGN_UV_NFD_all = pd.read_csv(f'AGN_Quantifying_Change_Sample_{my_sample}_UV_all.csv')
    AGN_UV_NFD_all = AGN_UV_NFD_all.dropna(subset=[AGN_UV_NFD_all.columns[1]])
    AGN_UV_NFD = AGN_UV_NFD_all.iloc[:, 1].tolist()
    AGN_UV_NFD_unc = AGN_UV_NFD_all.iloc[:, 2].tolist()

    median_AGN_UV_NFD = np.nanmedian(AGN_UV_NFD)
    median_AGN_UV_NFD_unc = np.nanmedian(AGN_UV_NFD_unc)
    three_sigma_UV_NFD = median_AGN_UV_NFD + 3*median_AGN_UV_NFD_unc

    AGN_UV_NFD_binsize = (np.nanmax(AGN_UV_NFD) - np.nanmin(AGN_UV_NFD))/20
    AGN_bins_UV_NFD = np.arange(np.nanmin(AGN_UV_NFD), np.nanmax(AGN_UV_NFD) + AGN_UV_NFD_binsize, AGN_UV_NFD_binsize)
    x_start = median_AGN_UV_NFD - 3*median_AGN_UV_NFD_unc
    x_end = median_AGN_UV_NFD + 3*median_AGN_UV_NFD_unc
    counts, bin_edges = np.histogram(AGN_UV_NFD, bins=AGN_bins_UV_NFD)
    height = max(counts)/2
    # print(sum(counts))


    CLAGN_UV_NFD_all = pd.read_csv('CLAGN_Quantifying_Change_UV_all.csv')
    CLAGN_UV_NFD_all = CLAGN_UV_NFD_all.dropna(subset=[CLAGN_UV_NFD_all.columns[1]])
    CLAGN_UV_NFD = CLAGN_UV_NFD_all.iloc[:, 1].tolist()
    CLAGN_UV_NFD_unc = CLAGN_UV_NFD_all.iloc[:, 2].tolist()

    median_CLAGN_UV_NFD = np.nanmedian(CLAGN_UV_NFD)
    median_CLAGN_UV_NFD_unc = np.nanmedian(CLAGN_UV_NFD_unc)

    CLAGN_UV_NFD_binsize = (np.nanmax(CLAGN_UV_NFD) - np.nanmin(CLAGN_UV_NFD))/20
    CLAGN_bins_UV_NFD = np.arange(np.nanmin(CLAGN_UV_NFD), np.nanmax(CLAGN_UV_NFD) + CLAGN_UV_NFD_binsize, CLAGN_UV_NFD_binsize)
    # counts, bin_edges = np.histogram(CLAGN_UV_NFD, bins=CLAGN_bins_UV_NFD)
    # print(sum(counts))

    a = 0
    for my_NFD in AGN_UV_NFD:
        if my_NFD > three_sigma_UV_NFD:
            a += 1
    
    b = 0
    for my_NFD in CLAGN_UV_NFD:
        if my_NFD > three_sigma_UV_NFD:
            b += 1
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)  
    ax1.hist(AGN_UV_NFD, bins=AGN_bins_UV_NFD, color='blue', edgecolor='black', label='Non-CL AGN Control Sample')
    ax1.axvline(median_AGN_UV_NFD, linewidth=2, linestyle='-', color='black', label=f'Non-CL AGN Median UV NFD = {median_AGN_UV_NFD:.2f}')
    ax1.axvline(three_sigma_UV_NFD, linewidth=2, linestyle='--', color='black', label=f'{a/len(AGN_UV_NFD)*100:.1f}% > Threshold = {three_sigma_UV_NFD:.2f}')
    ax1.plot((median_AGN_UV_NFD, x_end), (height-2, height-2), linewidth=2, color='peru')
    ax1.text(x_end+0.1, height-2, f'3X Median UV NFD Uncertainty = {3*median_AGN_UV_NFD_unc:.2f}', 
            ha='left', va='center', fontsize=16, color='peru')
    ax1.set_ylabel('Non-CL AGN Frequency', color='black', fontsize=18)
    ax1.text(2.25, 75, f'Non-CL AGN', ha='left', va='center', fontsize=16, color='blue')
    ax1.tick_params(axis='both', labelsize=18, length=8, width=2)
    ax1.legend(loc='upper right', fontsize=16)

    ax2.hist(CLAGN_UV_NFD, bins=CLAGN_bins_UV_NFD, color='red', edgecolor='black', label='Guo CLAGN Sample')
    ax2.axvline(median_CLAGN_UV_NFD, linewidth=2, linestyle='-', color='black', label=f'CLAGN Median UV NFD = {median_CLAGN_UV_NFD:.2f}')
    ax2.axvline(three_sigma_UV_NFD, linewidth=2, linestyle='--', color='black', label=f'{b/len(CLAGN_UV_NFD)*100:.1f}% > Threshold = {three_sigma_UV_NFD:.2f}')
    ax2.set_xlabel('UV NFD', fontsize=18)
    ax2.set_ylabel('CLAGN Frequency', color='black', fontsize=18)
    ax2.text(2.25, 6, f'CLAGN', ha='left', va='center', fontsize=16, color='red')
    ax2.tick_params(axis='both', labelsize=18, length=8, width=2)
    ax2.legend(loc='upper right', fontsize=16)

    plt.suptitle('UV NFD Distribution - Non-CL AGN Control vs Guo CLAGN', fontsize=22)
    plt.tight_layout()
    plt.show()


# # # #Creating a 2d plot of z score vs 2nd lowest flux:
if zs_W1_low == 1:
    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_W1_zscore_mean, AGN_W1_low_flux, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_W1_zscore_mean, CLAGN_W1_low_flux, color='red',  label='CLAGN')
    plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2, label = 'Threshold')
    plt.axhline(y=bright_dim_W1, color='black', linestyle='-', linewidth=2, label = f'Bright/Dim = {bright_dim_W1}')
    plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
    plt.ylim(0, 1.05*max(CLAGN_W1_low_flux+AGN_W1_low_flux))
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.xlabel("W1 Z-Score", fontsize = 26)
    plt.ylabel("W1 Min Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$", fontsize = 26)
    plt.title("W1 Min Flux vs W1 Z-Score", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.tight_layout()
    plt.show()


if zs_W2_low == 1:
    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_W2_zscore_mean, AGN_W2_low_flux, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_W2_zscore_mean, CLAGN_W2_low_flux, color='red',  label='CLAGN')
    plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2, label = 'Threshold')
    plt.axhline(y=bright_dim_W2, color='black', linestyle='-', linewidth=2, label = f'Bright/Dim = {bright_dim_W2}')
    plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
    plt.ylim(0, 1.05*max(CLAGN_W2_low_flux+AGN_W2_low_flux))
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.xlabel("W2 Z-Score", fontsize = 26)
    plt.ylabel("W2 Min Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$", fontsize = 26)
    plt.title("W2 Min Flux vs W2 Z-Score", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.tight_layout()
    plt.show()


# # #Creating a 2d plot of norm flux diff vs 2nd lowest flux:
if NFD_W1_low == 1:
    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_W1_NFD, AGN_W1_low_flux, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_W1_NFD, CLAGN_W1_low_flux, color='red',  label='CLAGN')
    if main_MIR_NFD_hist_bright_dim == 1:
        plt.axvline(x=three_sigma_norm_flux_diff_bright, color='black', linestyle='--', linewidth=2, label = f'Bright Threshold = {three_sigma_norm_flux_diff_bright:.1f}')
        plt.axvline(x=three_sigma_norm_flux_diff_dim, color='black', linestyle=':', linewidth=2, label = f'Dim Threshold = {three_sigma_norm_flux_diff_dim:.1f}')
    plt.axhline(y=bright_dim_W1, color='black', linestyle='-', linewidth=2, label = f'Bright/Dim = {bright_dim_W1}')
    plt.xlim(0, 1.05*max(CLAGN_norm_flux_diff+AGN_norm_flux_diff))
    plt.ylim(0, 1.05*max(CLAGN_W1_low_flux+AGN_W1_low_flux))
    plt.tick_params(axis='both', labelsize=22, length=8, width=2)
    plt.xlabel("W1 NFD", fontsize = 22)
    plt.ylabel("W1 Min Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$", fontsize = 22)
    plt.title("W1 Min Flux vs W1 NFD", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.tight_layout()
    plt.show()

if NFD_W2_low == 1:
    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_W2_NFD, AGN_W2_low_flux, color='blue', label='Non-CL AGN')
    plt.scatter(CLAGN_W2_NFD, CLAGN_W2_low_flux, color='red',  label='CLAGN')
    plt.axvline(x=three_sigma_norm_flux_diff, color='black', linestyle='--', linewidth=2, label = 'Threshold')
    plt.axhline(y=bright_dim_W2, color='black', linestyle='-', linewidth=2, label = f'Bright/Dim = {bright_dim_W2}')
    plt.xlim(0, 1.05*max(CLAGN_norm_flux_diff+AGN_norm_flux_diff))
    plt.ylim(0, 1.05*max(CLAGN_W2_low_flux+AGN_W2_low_flux))
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.xlabel("W2 NFD", fontsize = 26)
    plt.ylabel("W2 Min Flux / $10^{-17}$ergs $s^{-1}cm^{-2}Å^{-1}$", fontsize = 26)
    plt.title("W2 Min Flux vs W2 NFD", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.tight_layout()
    plt.show()


# # # #Creating a 2d plot of redshift vs z score:
# plt.figure(figsize=(12, 7))
# plt.scatter(CLAGN_zscores, CLAGN_redshifts_analysis, color='red', s=100, label='CLAGN')
# plt.scatter(AGN_zscores, AGN_redshifts_analysis, color='blue', label='Non-CL AGN')
# plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2)
# plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
# plt.ylim(0, 1.05*max(CLAGN_redshifts_analysis+AGN_redshifts_analysis))
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
# fit_params_CLAGN = np.polyfit(inverse_CLAGN_zscores, CLAGN_redshifts_analysis, 1)  # Degree 1 for a linear fit
# slope_CLAGN, intercept_CLAGN = fit_params_CLAGN
# y_fit_CLAGN = slope_CLAGN*np.array(inverse_CLAGN_zscores)+intercept_CLAGN
# fit_params_AGN = np.polyfit(inverse_AGN_zscores, AGN_redshifts_analysis, 1)  # Degree 1 for a linear fit
# slope_AGN, intercept_AGN = fit_params_AGN
# y_fit_AGN = slope_AGN*np.array(inverse_AGN_zscores)+intercept_AGN
# combined_zscores = inverse_CLAGN_zscores+inverse_AGN_zscores
# combined_redshifts = CLAGN_redshifts_analysis+AGN_redshifts_analysis
# fit_params_both = np.polyfit(combined_zscores, combined_redshifts, 1)  # Degree 1 for a linear fit
# slope_both, intercept_both = fit_params_both
# y_fit_both = slope_both*np.array(combined_zscores)+intercept_both

# plt.figure(figsize=(12, 7))
# plt.scatter(inverse_CLAGN_zscores, CLAGN_redshifts_analysis, color='red', s=100, label='CLAGN')
# plt.scatter(inverse_AGN_zscores, AGN_redshifts_analysis, color='blue', label='Non-CL AGN')
# plt.plot(inverse_CLAGN_zscores, y_fit_CLAGN, color="red", label=f"CLAGN: y={slope_CLAGN:.2f}x+{intercept_CLAGN:.2f}")
# plt.plot(inverse_AGN_zscores, y_fit_AGN, color="blue", label=f"AGN: y={slope_AGN:.2f}x+{intercept_AGN:.2f}")
# plt.plot(combined_zscores, y_fit_both, color="black", label=f"Comb: y={slope_both:.2f}x+{intercept_both:.2f}")
# plt.xlim(0, 1.05*max(inverse_CLAGN_zscores+inverse_AGN_zscores))
# plt.ylim(0, 1.05*max(CLAGN_redshifts_analysis+AGN_redshifts_analysis))
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
# plt.scatter(CLAGN_redshifts_analysis, CLAGN_norm_flux_diff, color='red',  label='CLAGN')
# plt.scatter(AGN_redshifts_analysis, AGN_norm_flux_diff, color='blue',  label='Non-CL AGN')
# plt.axvline(x=three_sigma_norm_flux_diff, color='black', linestyle='--', linewidth=2)
# plt.xlim(0, 1.05*max(CLAGN_norm_flux_diff+AGN_norm_flux_diff))
# plt.ylim(0, 1.05*max(CLAGN_redshifts_analysis+AGN_redshifts_analysis))
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
# plt.scatter(CLAGN_redshifts_analysis, CLAGN_W1_low_flux, s=100, color='red',  label='CLAGN')
# plt.scatter(AGN_redshifts_analysis, AGN_W1_low_flux, color='blue',  label='Non-CL AGN')
# plt.xlim(0, 1.05*max(CLAGN_redshifts_analysis+AGN_redshifts_analysis))
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
# plt.scatter(CLAGN_W1_median_flux_unc, CLAGN_redshifts_analysis, color='red',  label='CLAGN')
# plt.scatter(AGN_W1_median_flux_unc, AGN_redshifts_analysis, color='blue',  label='Non-CL AGN')
# plt.xlim(0, 1.05*max(CLAGN_W1_median_flux_unc+AGN_W1_median_flux_unc))
# plt.ylim(0, 1.05*max(CLAGN_redshifts_analysis+AGN_redshifts_analysis))
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
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
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
    elif brightness == 2:
        plt.text(0.99, 0.31, f'CLAGN Median W1 NFD = {CLAGN_median_W1_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.19, f'AGN Median W1 NFD = {AGN_median_W1_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.37, f'CLAGN Median W2 NFD = {CLAGN_median_W2_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.25, f'AGN Median W2 NFD = {AGN_median_W2_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.legend(loc = 'upper left', fontsize=22)
    plt.tight_layout()
    plt.show()


if W1_vs_W2_NFD_direction == 1:
    AGN_plot = 1

    CLAGN_names_analysis = np.array(CLAGN_names_analysis)
    CLAGN_W1_min_mjd = np.array(CLAGN_W1_min_mjd)
    CLAGN_W1_max_mjd = np.array(CLAGN_W1_max_mjd)
    CLAGN_W1_NFD = np.array(CLAGN_W1_NFD)
    # Find indices where min_mjd > max_mjd
    CLAGN_invert_indices_W1 = CLAGN_W1_min_mjd > CLAGN_W1_max_mjd
    CLAGN_names = CLAGN_quantifying_change_data.iloc[:, 0].tolist()
    # Make corresponding W1_NFD values negative
    CLAGN_W1_NFD[CLAGN_invert_indices_W1] *= -1
    CLAGN_W1_NFD = CLAGN_W1_NFD.tolist()
    invert_list_W1 = CLAGN_names_analysis[CLAGN_invert_indices_W1]
    not_invert_list_W1 = np.setdiff1d(CLAGN_names_analysis, invert_list_W1)
    print(f'CLAGN with min flux AFTER max flux W1 = {invert_list_W1}')
    print(f'CLAGN with min flux BEFORE max flux W1 = {not_invert_list_W1}')

    CLAGN_W2_min_mjd = np.array(CLAGN_W2_min_mjd)
    CLAGN_W2_max_mjd = np.array(CLAGN_W2_max_mjd)
    CLAGN_W2_NFD = np.array(CLAGN_W2_NFD)
    # Find indices where min_mjd > max_mjd
    CLAGN_invert_indices_W2 = CLAGN_W2_min_mjd > CLAGN_W2_max_mjd
    # Make corresponding W2_NFD values negative
    CLAGN_W2_NFD[CLAGN_invert_indices_W2] *= -1
    CLAGN_W2_NFD = CLAGN_W2_NFD.tolist()
    invert_list_W2 = CLAGN_names_analysis[CLAGN_invert_indices_W2]
    not_invert_list_W2 = np.setdiff1d(CLAGN_names_analysis, invert_list_W2)
    print(f'CLAGN with min flux AFTER max flux W2 = {invert_list_W2}')
    print(f'CLAGN with min flux BEFORE max flux W2 = {not_invert_list_W2}')

    if AGN_plot == 1:
        AGN_W1_min_mjd = np.array(AGN_W1_min_mjd)
        AGN_W1_max_mjd = np.array(AGN_W1_max_mjd)
        AGN_W1_NFD = np.array(AGN_W1_NFD)
        # Find indices where min_mjd > max_mjd
        AGN_invert_indices_W1 = AGN_W1_min_mjd > AGN_W1_max_mjd
        # Make corresponding W1_NFD values negative
        # AGN_W1_NFD[AGN_invert_indices_W1] *= -1
        AGN_W1_NFD = AGN_W1_NFD.tolist()

        AGN_W2_min_mjd = np.array(AGN_W2_min_mjd)
        AGN_W2_max_mjd = np.array(AGN_W2_max_mjd)
        AGN_W2_NFD = np.array(AGN_W2_NFD)
        # Find indices where min_mjd > max_mjd
        AGN_invert_indices_W2 = AGN_W2_min_mjd > AGN_W2_max_mjd
        # Make corresponding W2_NFD values negative
        # AGN_W2_NFD[AGN_invert_indices_W2] *= -1
        AGN_W2_NFD = AGN_W2_NFD.tolist()

        max_W1 = np.nanmax(CLAGN_W1_NFD+AGN_W1_NFD)
        min_W1 = np.nanmin(CLAGN_W1_NFD+AGN_W1_NFD)
        max_W2 = np.nanmax(CLAGN_W2_NFD+AGN_W2_NFD)
        min_W2 = np.nanmin(CLAGN_W2_NFD+AGN_W2_NFD)
        AGN_median_W1_NFD = np.nanmedian(AGN_W1_NFD)
        AGN_median_W2_NFD = np.nanmedian(AGN_W2_NFD)
    elif AGN_plot == 0:
        max_W1 = np.nanmax(CLAGN_W1_NFD)
        min_W1 = np.nanmin(CLAGN_W1_NFD)
        max_W2 = np.nanmax(CLAGN_W2_NFD)
        min_W2 = np.nanmin(CLAGN_W2_NFD)

    CLAGN_median_W1_NFD = np.nanmedian(CLAGN_W1_NFD)
    CLAGN_median_W2_NFD = np.nanmedian(CLAGN_W2_NFD)
    plt.figure(figsize=(12, 7))
    if AGN_plot == 1:
        plt.scatter(AGN_W1_NFD, AGN_W2_NFD, color='blue',  label='Non-CL AGN')
    plt.scatter(CLAGN_W1_NFD, CLAGN_W2_NFD, s=100, color='red',  label='CLAGN')
    plt.axvline(0, color='black', linestyle=':')
    plt.axhline(0, color='black', linestyle=':')
    plt.xlim(1.1*min_W1, 1.1*max_W1)
    plt.ylim(1.1*min_W2, 1.1*max_W2)
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.xlabel("W1 NFD", fontsize = 24)
    plt.ylabel("W2 NFD", fontsize = 24)
    if turn_on_off == 0:
        plt.title("W1 NFD vs W2 NFD (turn-off CLAGN)", fontsize = 24)
    elif turn_on_off == 1:
        plt.title("W1 NFD vs W2 NFD (turn-on CLAGN)", fontsize = 24)
    elif turn_on_off == 2:
        plt.title("W1 NFD vs W2 NFD", fontsize = 24)
    ax = plt.gca()
    if brightness == 0:
        plt.text(0.99, 0.85, f'CLAGN Median W1 NFD = {CLAGN_median_W1_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.91, f'CLAGN Median W2 NFD = {CLAGN_median_W2_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        if AGN_plot == 1:
            plt.text(0.99, 0.73, f'AGN Median W1 NFD = {AGN_median_W1_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
            plt.text(0.99, 0.79, f'AGN Median W2 NFD = {AGN_median_W2_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.legend(loc = 'lower right', fontsize=22)
    elif brightness == 1:
        plt.text(0.99, 0.31, f'CLAGN Median W1 NFD = {CLAGN_median_W1_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.37, f'CLAGN Median W2 NFD = {CLAGN_median_W2_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        if AGN_plot == 1:
            plt.text(0.99, 0.19, f'AGN Median W1 NFD = {AGN_median_W1_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
            plt.text(0.99, 0.25, f'AGN Median W2 NFD = {AGN_median_W2_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.legend(loc = 'upper left', fontsize=22)
    elif brightness == 2:
        plt.text(0.04, 0.40, f'CLAGN Median W1 NFD = {CLAGN_median_W1_NFD:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.04, 0.46, f'CLAGN Median W2 NFD = {CLAGN_median_W2_NFD:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        if AGN_plot == 1:
            plt.text(0.99, 0.19, f'AGN Median W1 NFD = {AGN_median_W1_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
            plt.text(0.99, 0.25, f'AGN Median W2 NFD = {AGN_median_W2_NFD:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.legend(loc = 'best', fontsize=22)
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
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
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


if W1_vs_W2_Zs_direction == 1:
    AGN_plot = 1

    CLAGN_W1_min_mjd = np.array(CLAGN_W1_min_mjd)
    CLAGN_W1_max_mjd = np.array(CLAGN_W1_max_mjd)
    CLAGN_W1_zscore_mean = np.array(CLAGN_W1_zscore_mean)
    # Find indices where min_mjd > max_mjd
    invert_indices_W1 = CLAGN_W1_min_mjd > CLAGN_W1_max_mjd
    CLAGN_W1_zscore_mean[invert_indices_W1] *= -1
    CLAGN_W1_zscore_mean = CLAGN_W1_zscore_mean.tolist()

    CLAGN_W2_min_mjd = np.array(CLAGN_W2_min_mjd)
    CLAGN_W2_max_mjd = np.array(CLAGN_W2_max_mjd)
    CLAGN_W2_zscore_mean = np.array(CLAGN_W2_zscore_mean)
    # Find indices where min_mjd > max_mjd
    invert_indices_W2 = CLAGN_W2_min_mjd > CLAGN_W2_max_mjd
    CLAGN_W2_zscore_mean[invert_indices_W2] *= -1
    CLAGN_W2_zscore_mean = CLAGN_W2_zscore_mean.tolist()

    if AGN_plot == 1:
        AGN_W1_min_mjd = np.array(AGN_W1_min_mjd)
        AGN_W1_max_mjd = np.array(AGN_W1_max_mjd)
        AGN_W1_zscore_mean = np.array(AGN_W1_zscore_mean)
        # Find indices where min_mjd > max_mjd
        invert_indices_W1 = AGN_W1_min_mjd > AGN_W1_max_mjd
        # AGN_W1_zscore_mean[invert_indices_W1] *= -1
        AGN_W1_zscore_mean = AGN_W1_zscore_mean.tolist()

        AGN_W2_min_mjd = np.array(AGN_W2_min_mjd)
        AGN_W2_max_mjd = np.array(AGN_W2_max_mjd)
        AGN_W2_zscore_mean = np.array(AGN_W2_zscore_mean)
        # Find indices where min_mjd > max_mjd
        invert_indices_W2 = AGN_W2_min_mjd > AGN_W2_max_mjd
        # AGN_W2_zscore_mean[invert_indices_W2] *= -1
        AGN_W2_zscore_mean = AGN_W2_zscore_mean.tolist()
    
        max_W1 = np.nanmax(CLAGN_W1_zscore_mean+AGN_W1_zscore_mean)
        min_W1 = np.nanmin(CLAGN_W1_zscore_mean+AGN_W1_zscore_mean)
        max_W2 = np.nanmax(CLAGN_W2_zscore_mean+AGN_W2_zscore_mean)
        min_W2 = np.nanmin(CLAGN_W2_zscore_mean+AGN_W2_zscore_mean)

        AGN_median_W1_zs = np.nanmedian(AGN_W1_zscore_mean)
        AGN_median_W2_zs = np.nanmedian(AGN_W2_zscore_mean)

    elif AGN_plot == 0:
        max_W1 = np.nanmax(CLAGN_W1_zscore_mean)
        min_W1 = np.nanmin(CLAGN_W1_zscore_mean)
        max_W2 = np.nanmax(CLAGN_W2_zscore_mean)
        min_W2 = np.nanmin(CLAGN_W2_zscore_mean)

    CLAGN_median_W1_zs = np.nanmedian(CLAGN_W1_zscore_mean)
    CLAGN_median_W2_zs = np.nanmedian(CLAGN_W2_zscore_mean)
    x = np.linspace(0, min([max_W1, max_W2]), 100)
    plt.figure(figsize=(12, 7))
    if AGN_plot == 1:
        plt.scatter(AGN_W1_zscore_mean, AGN_W2_zscore_mean, color='blue',  label='Non-CL AGN')
    plt.scatter(CLAGN_W1_zscore_mean, CLAGN_W2_zscore_mean, s=100, color='red',  label='CLAGN')
    plt.axhline(0, color='black', linestyle=':')
    plt.axvline(0, color='black', linestyle=':')
    plt.xlim(1.1*min_W1, 1.1*max_W1)
    plt.ylim(1.1*min_W2, 1.5*max_W2)
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.xlabel("W1 Z-score", fontsize = 24)
    plt.ylabel("W2 Z-score", fontsize = 24)
    if turn_on_off == 0:
        plt.title("W1 Z-score vs W2 Z-score (turn-off CLAGN)", fontsize = 24)
    elif turn_on_off == 1:
        plt.title("W1 Z-score vs W2 Z-score (turn-on CLAGN)", fontsize = 24)
    elif turn_on_off == 2:
        plt.title("W1 Z-score vs W2 Z-score", fontsize = 24)
    plt.legend(loc = 'best', fontsize=22)
    ax = plt.gca()
    if brightness == 0:
        plt.text(0.99, 0.67, f'CLAGN Median W1 Z-score = {CLAGN_median_W1_zs:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.99, 0.73, f'CLAGN Median W2 Z-score = {CLAGN_median_W2_zs:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        if AGN_plot == 1:
            plt.text(0.99, 0.55, f'AGN Median W1 Z-score = {AGN_median_W1_zs:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
            plt.text(0.99, 0.61, f'AGN Median W2 Z-score = {AGN_median_W2_zs:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
        plt.legend(loc = 'lower right', fontsize=22)
    elif brightness == 1:
        plt.text(0.01, 0.85, f'CLAGN Median W1 Z-score = {CLAGN_median_W1_zs:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.01, 0.91, f'CLAGN Median W2 Z-score = {CLAGN_median_W2_zs:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        if AGN_plot == 1:
            plt.text(0.01, 0.73, f'AGN Median W1 Z-score = {AGN_median_W1_zs:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
            plt.text(0.01, 0.79, f'AGN Median W2 Z-score = {AGN_median_W2_zs:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        plt.legend(loc = 'lower right', fontsize=22)
    elif brightness == 2:
        plt.text(0.20, 0.06, f'CLAGN Median W1 Z-score = {CLAGN_median_W1_zs:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        plt.text(0.20, 0.12, f'CLAGN Median W2 Z-score = {CLAGN_median_W2_zs:.1f}', fontsize = 25, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        if AGN_plot == 1:
            plt.text(0.99, 0.55, f'AGN Median W1 Z-score = {AGN_median_W1_zs:.1f}', fontsize = 25, horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
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
    threshold_CLAGN = 9
    CLAGN_mod_dev_list_elim = [x for x in CLAGN_mod_dev_list if abs(x) > threshold_CLAGN]
    percentage_elim_CLAGN = len(CLAGN_mod_dev_list_elim)/(len(CLAGN_mod_dev_list_elim)+len(CLAGN_mod_dev_list))*100

    #CLAGN
    # CLAGN_mod_dev_list = [x for x in CLAGN_mod_dev_list if abs(x) <= threshold_CLAGN]
    median_mod_dev = np.median(CLAGN_mod_dev_list)
    mod_dev_binsize = (max(CLAGN_mod_dev_list)-min(CLAGN_mod_dev_list))/250 #250 bins
    bins_mod_dev = np.arange(min(CLAGN_mod_dev_list), max(CLAGN_mod_dev_list) + 5*mod_dev_binsize, mod_dev_binsize)
    plt.figure(figsize=(12,7))
    plt.hist(CLAGN_mod_dev_list, bins=bins_mod_dev, color='darkorange', edgecolor='black', label=f'binsize = {mod_dev_binsize:.2f}')
    plt.axvline(median_mod_dev, linewidth=2, linestyle='--', color='black', label = f'Median = {median_mod_dev:.2f}')
    plt.axvline(threshold_CLAGN, linewidth=2, linestyle='-', color='black', label = f'Threshold = {threshold_CLAGN} eliminates {percentage_elim_CLAGN:.3f}% of non-CL AGN data')
    plt.yscale('log')
    plt.xlabel('Modified Deviation')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of CLAGN Modified Deviation Values < 100 from {len(CLAGN_mod_dev_list)} Observations')
    plt.legend(loc='upper right')
    plt.show()


    #Non-CL AGN
    threshold_AGN = 25
    AGN_mod_dev_list_elim = [x for x in AGN_mod_dev_list if abs(x) > threshold_AGN]
    percentage_elim_AGN = len(AGN_mod_dev_list_elim)/(len(AGN_mod_dev_list_elim)+len(AGN_mod_dev_list))*100

    # AGN_mod_dev_list = [x for x in AGN_mod_dev_list if abs(x) <= threshold_AGN]
    AGN_mod_dev_list = [x for x in AGN_mod_dev_list if abs(x) <= 100]
    median_mod_dev = np.median(AGN_mod_dev_list)
    mod_dev_binsize = (max(AGN_mod_dev_list)-min(AGN_mod_dev_list))/250 #250 bins
    bins_mod_dev = np.arange(min(AGN_mod_dev_list), max(AGN_mod_dev_list) + 5*mod_dev_binsize, mod_dev_binsize)
    plt.figure(figsize=(12,7))
    plt.hist(AGN_mod_dev_list, bins=bins_mod_dev, color='darkorange', edgecolor='black', label=f'binsize = {mod_dev_binsize:.2f}')
    plt.axvline(median_mod_dev, linewidth=2, linestyle='--', color='black', label = f'Median = {median_mod_dev:.2f}')
    plt.axvline(threshold_AGN, linewidth=2, linestyle='-', color='black', label = f'Threshold = {threshold_AGN} eliminates {percentage_elim_AGN:.3f}% of non-CL AGN data')
    plt.yscale('log')
    plt.xlabel('Modified Deviation')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of AGN Modified Deviation Values < 100 from {len(AGN_mod_dev_list)} Observations')
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


if Modified_Dev_epochs_plot == 1:
    CLAGN_mod_dev_W1 = pd.read_csv('CLAGN_modified_deviation_epoch_measurements_W1.csv')
    CLAGN_mod_dev_list_W1 = CLAGN_mod_dev_W1.iloc[:, 0].tolist()
    CLAGN_mod_dev_W2 = pd.read_csv('CLAGN_modified_deviation_epoch_measurements_W2.csv')
    CLAGN_mod_dev_list_W2 = CLAGN_mod_dev_W2.iloc[:, 0].tolist()
    threshold_CLAGN = 10
    CLAGN_mod_dev_list_elim_W1 = [x for x in CLAGN_mod_dev_list_W1 if abs(x) > threshold_CLAGN]
    CLAGN_mod_dev_list_elim_W2 = [x for x in CLAGN_mod_dev_list_W2 if abs(x) > threshold_CLAGN]
    percentage_elim_CLAGN_W1 = len(CLAGN_mod_dev_list_elim_W1)/(len(CLAGN_mod_dev_list_elim_W1)+len(CLAGN_mod_dev_list_W1))*100
    percentage_elim_CLAGN_W2 = len(CLAGN_mod_dev_list_elim_W2)/(len(CLAGN_mod_dev_list_elim_W2)+len(CLAGN_mod_dev_list_W2))*100

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    # First Histogram (W1)
    median_mod_dev_W1 = np.median(CLAGN_mod_dev_list_W1)
    mod_dev_binsize_W1 = (max(CLAGN_mod_dev_list_W1) - min(CLAGN_mod_dev_list_W1)) / 250
    bins_mod_dev_W1 = np.arange(min(CLAGN_mod_dev_list_W1), max(CLAGN_mod_dev_list_W1) + 5 * mod_dev_binsize_W1, mod_dev_binsize_W1)

    axes[0].hist(CLAGN_mod_dev_list_W1, bins=bins_mod_dev_W1, color='blue', edgecolor='black', label=f'Binsize = {mod_dev_binsize_W1:.2f}')
    axes[0].axvline(median_mod_dev_W1, linewidth=2, linestyle='--', color='black', label=f'Median = {median_mod_dev_W1:.2f}')
    axes[0].axvline(threshold_CLAGN, linewidth=2, linestyle='-', color='black', label=f'Threshold = {threshold_CLAGN} eliminates {percentage_elim_CLAGN_W1:.3f}% of CLAGN W1 epochs')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Modified Deviation (W1)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distribution of CLAGN Modified Deviation Values (W1) from {len(CLAGN_mod_dev_list_W1)} Epochs')
    axes[0].legend(loc='upper right')

    # Second Histogram (W2)
    median_mod_dev_W2 = np.median(CLAGN_mod_dev_list_W2)
    mod_dev_binsize_W2 = (max(CLAGN_mod_dev_list_W2) - min(CLAGN_mod_dev_list_W2)) / 250
    bins_mod_dev_W2 = np.arange(min(CLAGN_mod_dev_list_W2), max(CLAGN_mod_dev_list_W2) + 5 * mod_dev_binsize_W2, mod_dev_binsize_W2)

    axes[1].hist(CLAGN_mod_dev_list_W2, bins=bins_mod_dev_W2, color='orange', edgecolor='black', label=f'Binsize = {mod_dev_binsize_W2:.2f}')
    axes[1].axvline(median_mod_dev_W2, linewidth=2, linestyle='--', color='black', label=f'Median = {median_mod_dev_W2:.2f}')
    axes[1].axvline(threshold_CLAGN, linewidth=2, linestyle='-', color='black', label=f'Threshold = {threshold_CLAGN} eliminates {percentage_elim_CLAGN_W2:.3f}% of CLAGN W2 epochs')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Modified Deviation (W2)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Distribution of CLAGN Modified Deviation Values (W2) from {len(CLAGN_mod_dev_list_W2)} Epochs')
    axes[1].legend(loc='upper right')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    # #Non-CL AGN
    AGN_mod_dev_W1 = pd.read_csv('AGN_modified_deviation_epoch_measurements_sample_1_W1.csv')
    AGN_mod_dev_list_W1 = AGN_mod_dev_W1.iloc[:, 0].tolist()
    AGN_mod_dev_W2 = pd.read_csv('AGN_modified_deviation_epoch_measurements_sample_1_W2.csv')
    AGN_mod_dev_list_W2 = AGN_mod_dev_W2.iloc[:, 0].tolist()
    threshold_AGN = 10
    AGN_mod_dev_list_elim_W1 = [x for x in AGN_mod_dev_list_W1 if abs(x) > threshold_AGN]
    AGN_mod_dev_list_elim_W2 = [x for x in AGN_mod_dev_list_W2 if abs(x) > threshold_AGN]
    percentage_elim_AGN_W1 = len(AGN_mod_dev_list_elim_W1)/(len(AGN_mod_dev_list_elim_W1)+len(AGN_mod_dev_list_W1))*100
    percentage_elim_AGN_W2 = len(AGN_mod_dev_list_elim_W2)/(len(AGN_mod_dev_list_elim_W2)+len(AGN_mod_dev_list_W2))*100

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    # First Histogram (W1)
    median_mod_dev_W1 = np.median(AGN_mod_dev_list_W1)
    mod_dev_binsize_W1 = (max(AGN_mod_dev_list_W1) - min(AGN_mod_dev_list_W1)) / 250
    bins_mod_dev_W1 = np.arange(min(AGN_mod_dev_list_W1), max(AGN_mod_dev_list_W1) + 5 * mod_dev_binsize_W1, mod_dev_binsize_W1)

    axes[0].hist(AGN_mod_dev_list_W1, bins=bins_mod_dev_W1, color='blue', edgecolor='black', label=f'Binsize = {mod_dev_binsize_W1:.2f}')
    axes[0].axvline(median_mod_dev_W1, linewidth=2, linestyle='--', color='black', label=f'Median = {median_mod_dev_W1:.2f}')
    axes[0].axvline(threshold_AGN, linewidth=2, linestyle='-', color='black', label=f'Threshold = {threshold_AGN} eliminates {percentage_elim_AGN_W1:.3f}% of non-CL AGN W1 epochs')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Modified Deviation (W1)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distribution of AGN Modified Deviation Values (W1) from {len(AGN_mod_dev_list_W1)} Epochs')
    axes[0].legend(loc='upper right')

    # Second Histogram (W2)
    median_mod_dev_W2 = np.median(AGN_mod_dev_list_W2)
    mod_dev_binsize_W2 = (max(AGN_mod_dev_list_W2) - min(AGN_mod_dev_list_W2)) / 250
    bins_mod_dev_W2 = np.arange(min(AGN_mod_dev_list_W2), max(AGN_mod_dev_list_W2) + 5 * mod_dev_binsize_W2, mod_dev_binsize_W2)

    axes[1].hist(AGN_mod_dev_list_W2, bins=bins_mod_dev_W2, color='orange', edgecolor='black', label=f'Binsize = {mod_dev_binsize_W2:.2f}')
    axes[1].axvline(median_mod_dev_W2, linewidth=2, linestyle='--', color='black', label=f'Median = {median_mod_dev_W2:.2f}')
    axes[1].axvline(threshold_AGN, linewidth=2, linestyle='-', color='black', label=f'Threshold = {threshold_AGN} eliminates {percentage_elim_AGN_W2:.3f}% of non-CL AGN W2 epochs')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Modified Deviation (W2)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Distribution of AGN Modified Deviation Values (W2) from {len(AGN_mod_dev_list_W2)} Epochs')
    axes[1].legend(loc='upper right')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


if Modified_Dev_vs_epoch_measurements_plot == 1:
    CLAGN_mod_dev_W1 = pd.read_csv('CLAGN_modified_deviation_epoch_measurements_W1.csv')
    CLAGN_mod_dev_list_W1 = CLAGN_mod_dev_W1.iloc[:, 0].tolist()
    CLAGN_epoch_measurements_list_W1 = CLAGN_mod_dev_W1.iloc[:, 1].tolist()
    CLAGN_mod_dev_W2 = pd.read_csv('CLAGN_modified_deviation_epoch_measurements_W2.csv')
    CLAGN_mod_dev_list_W2 = CLAGN_mod_dev_W2.iloc[:, 0].tolist()
    CLAGN_epoch_measurements_list_W2 = CLAGN_mod_dev_W2.iloc[:, 1].tolist()
    threshold_CLAGN = 10
    CLAGN_mod_dev_list_elim_W1 = [x for x in CLAGN_mod_dev_list_W1 if abs(x) > threshold_CLAGN]
    CLAGN_mod_dev_list_elim_W2 = [x for x in CLAGN_mod_dev_list_W2 if abs(x) > threshold_CLAGN]
    percentage_elim_CLAGN_W1 = len(CLAGN_mod_dev_list_elim_W1)/(len(CLAGN_mod_dev_list_elim_W1)+len(CLAGN_mod_dev_list_W1))*100
    percentage_elim_CLAGN_W2 = len(CLAGN_mod_dev_list_elim_W2)/(len(CLAGN_mod_dev_list_elim_W2)+len(CLAGN_mod_dev_list_W2))*100

    print(f'Median number of CLAGN W1 measurements in an epoch = {np.nanmean(CLAGN_epoch_measurements_list_W1)}')
    print(f'Median number of CLAGN W2 measurements in an epoch = {np.nanmean(CLAGN_epoch_measurements_list_W2)}')

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    # First Histogram (W1)
    axes[0].scatter(CLAGN_mod_dev_list_W1, CLAGN_epoch_measurements_list_W1, color='blue',  label='CLAGN W1')
    axes[0].axvline(threshold_CLAGN, linewidth=2, linestyle='-', color='black', label=f'Threshold = {threshold_CLAGN} eliminates {percentage_elim_CLAGN_W1:.3f}% of CLAGN W1 epochs')
    axes[0].set_xlabel('Modified Deviation (W1)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Epoch Measurements vs Modified Deviation - {len(CLAGN_mod_dev_list_W1)} Epochs')
    axes[0].legend(loc='upper right')

    # Second Histogram (W2)
    axes[1].scatter(CLAGN_mod_dev_list_W2, CLAGN_epoch_measurements_list_W2, color='orange',  label='CLAGN W2')
    axes[1].axvline(threshold_CLAGN, linewidth=2, linestyle='-', color='black', label=f'Threshold = {threshold_CLAGN} eliminates {percentage_elim_CLAGN_W2:.3f}% of CLAGN W2 epochs')
    axes[1].set_xlabel('Modified Deviation (W2)')
    axes[1].set_ylabel('Epoch Measurements')
    axes[1].set_title(f'Epoch Measurements vs Modified Deviation - {len(CLAGN_mod_dev_list_W2)} Epochs')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()


    # #Non-CL AGN
    AGN_mod_dev_W1 = pd.read_csv('AGN_modified_deviation_epoch_measurements_sample_1_W1.csv')
    AGN_mod_dev_list_W1 = AGN_mod_dev_W1.iloc[:, 0].tolist()
    AGN_epoch_measurements_list_W1 = AGN_mod_dev_W1.iloc[:, 1].tolist()
    AGN_mod_dev_W2 = pd.read_csv('AGN_modified_deviation_epoch_measurements_sample_1_W2.csv')
    AGN_mod_dev_list_W2 = AGN_mod_dev_W2.iloc[:, 0].tolist()
    AGN_epoch_measurements_list_W2 = AGN_mod_dev_W2.iloc[:, 1].tolist()
    threshold_AGN = 10
    AGN_mod_dev_list_elim_W1 = [x for x in AGN_mod_dev_list_W1 if abs(x) > threshold_AGN]
    AGN_mod_dev_list_elim_W2 = [x for x in AGN_mod_dev_list_W2 if abs(x) > threshold_AGN]
    percentage_elim_AGN_W1 = len(AGN_mod_dev_list_elim_W1)/(len(AGN_mod_dev_list_elim_W1)+len(AGN_mod_dev_list_W1))*100
    percentage_elim_AGN_W2 = len(AGN_mod_dev_list_elim_W2)/(len(AGN_mod_dev_list_elim_W2)+len(AGN_mod_dev_list_W2))*100

    print(f'Median number of AGN W1 measurements in an epoch = {np.nanmean(AGN_epoch_measurements_list_W1)}')
    print(f'Median number of AGN W2 measurements in an epoch = {np.nanmean(AGN_epoch_measurements_list_W2)}')

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    # First Histogram (W1)
    axes[0].scatter(AGN_mod_dev_list_W1, AGN_epoch_measurements_list_W1, color='blue',  label='AGN')
    axes[0].axvline(threshold_AGN, linewidth=2, linestyle='-', color='black', label=f'Threshold = {threshold_AGN} eliminates {percentage_elim_AGN_W1:.3f}% of non-CL AGN W1 epochs')
    axes[0].set_xlabel('Modified Deviation (W1)')
    axes[0].set_ylabel('Epoch Measurements')
    axes[0].set_title(f'Epoch Measurements vs Modified Deviation - {len(AGN_mod_dev_list_W1)} Epochs')
    axes[0].legend(loc='upper right')

    # Second Histogram (W2)
    axes[1].scatter(AGN_mod_dev_list_W2, AGN_epoch_measurements_list_W2, color='orange',  label='AGN')
    axes[1].axvline(threshold_AGN, linewidth=2, linestyle='-', color='black', label=f'Threshold = {threshold_AGN} eliminates {percentage_elim_AGN_W2:.3f}% of non-CL AGN W2 epochs')
    axes[1].set_xlabel('Modified Deviation (W2)')
    axes[1].set_ylabel('Epoch Measurments')
    axes[1].set_title(f'Epoch Measurements vs Modified Deviation - {len(AGN_mod_dev_list_W2)} Epochs')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()


if Mean_unc_vs_epoch_meas_results == 1:
    CLAGN_data_W1 = pd.read_csv(f'CLAGN_mean_unc_vs_epoch_meas_W1.csv')
    CLAGN_epoch_measurements_list_W1 = CLAGN_data_W1.iloc[:, 1].tolist()
    CLAGN_mean_unc_W1 = CLAGN_data_W1[CLAGN_data_W1.iloc[:, 0] > 0.5]
    CLAGN_epoch_measurements_mean_unc_W1 = CLAGN_mean_unc_W1.iloc[:, 1].tolist()
    CLAGN_median_unc_W1 = CLAGN_data_W1[CLAGN_data_W1.iloc[:, 0] < 0.5]
    CLAGN_epoch_measurements_median_unc_W1 = CLAGN_median_unc_W1.iloc[:, 1].tolist()

    a = 0
    for measurements_in_epoch in CLAGN_epoch_measurements_list_W1:
        if measurements_in_epoch <= 4:
            a += 1

    CLAGN_data_W2 = pd.read_csv(f'CLAGN_mean_unc_vs_epoch_meas_W2.csv')
    CLAGN_epoch_measurements_list_W2 = CLAGN_data_W2.iloc[:, 1].tolist()
    CLAGN_mean_unc_W2 = CLAGN_data_W2[CLAGN_data_W2.iloc[:, 0] > 0.5]
    CLAGN_epoch_measurements_mean_unc_W2 = CLAGN_mean_unc_W2.iloc[:, 1].tolist()
    CLAGN_median_unc_W2 = CLAGN_data_W2[CLAGN_data_W2.iloc[:, 0] < 0.5]
    CLAGN_epoch_measurements_median_unc_W2 = CLAGN_median_unc_W2.iloc[:, 1].tolist()

    b = 0
    for measurements_in_epoch in CLAGN_epoch_measurements_list_W2:
        if measurements_in_epoch <= 4:
            b += 1

    print(f'CLAGN - % of epochs where mean unc used (W1) = {len(CLAGN_mean_unc_W1)/len(CLAGN_data_W1)*100:.2f}%')
    print(f'CLAGN - % of epochs where mean unc used (W2) = {len(CLAGN_mean_unc_W2)/len(CLAGN_data_W2)*100:.2f}%')
    print(f'CLAGN - % of epochs with 4 or less data points (W1) = {a/len(CLAGN_data_W1)*100:.2f}%')
    print(f'CLAGN - % of epochs with 4 or less data points (W2) = {b/len(CLAGN_data_W2)*100:.2f}%')
    print(f'CLAGN - Median number of measurements in epoch (W2) = {np.nanmedian(CLAGN_data_W2.iloc[:, 1].tolist())}')
    print(f'CLAGN - Median number of measurements in an epoch when mean unc used (W1) = {np.nanmedian(CLAGN_epoch_measurements_mean_unc_W1)}')
    print(f'CLAGN - Median number of measurements in an epoch when median unc used (W1) = {np.nanmedian(CLAGN_epoch_measurements_median_unc_W1)}')
    print(f'CLAGN - Median number of measurements in an epoch when mean unc used (W2) = {np.nanmedian(CLAGN_epoch_measurements_mean_unc_W2)}')
    print(f'CLAGN - Median number of measurements in an epoch when median unc used (W2) = {np.nanmedian(CLAGN_epoch_measurements_median_unc_W2)}')

    AGN_data_W1 = pd.read_csv(f'AGN_mean_unc_vs_epoch_meas_Sample_{my_sample}_W1.csv')
    AGN_epoch_measurements_list_W1 = AGN_data_W1.iloc[:, 1].tolist()
    AGN_mean_unc_W1 = AGN_data_W1[AGN_data_W1.iloc[:, 0] > 0.5]
    AGN_epoch_measurements_mean_unc_W1 = AGN_mean_unc_W1.iloc[:, 1].tolist()
    AGN_median_unc_W1 = AGN_data_W1[AGN_data_W1.iloc[:, 0] < 0.5]
    AGN_epoch_measurements_median_unc_W1 = AGN_median_unc_W1.iloc[:, 1].tolist()

    c = 0
    for measurements_in_epoch in AGN_epoch_measurements_list_W1:
        if measurements_in_epoch <= 4:
            c += 1

    AGN_data_W2 = pd.read_csv(f'AGN_mean_unc_vs_epoch_meas_Sample_{my_sample}_W2.csv')
    AGN_epoch_measurements_list_W2 = AGN_data_W2.iloc[:, 1].tolist()
    AGN_mean_unc_W2 = AGN_data_W2[AGN_data_W2.iloc[:, 0] > 0.5]
    AGN_epoch_measurements_mean_unc_W2 = AGN_mean_unc_W2.iloc[:, 1].tolist()
    AGN_median_unc_W2 = AGN_data_W2[AGN_data_W2.iloc[:, 0] < 0.5]
    AGN_epoch_measurements_median_unc_W2 = AGN_median_unc_W2.iloc[:, 1].tolist()

    d = 0
    for measurements_in_epoch in AGN_epoch_measurements_list_W2:
        if measurements_in_epoch <= 4:
            d += 1

    print(f'AGN - % of epochs where mean unc used (W1) = {len(AGN_mean_unc_W1)/len(AGN_data_W1)*100:.2f}%')
    print(f'AGN - % of epochs where mean unc used (W2) = {len(AGN_mean_unc_W2)/len(AGN_data_W2)*100:.2f}%')
    print(f'AGN - % of epochs with 4 or less data points (W1) = {c/len(AGN_data_W1)*100:.2f}%')
    print(f'AGN - % of epochs with 4 or less data points (W2) = {d/len(AGN_data_W2)*100:.2f}%')
    print(f'AGN - Median number of measurements in epoch (W1) = {np.nanmedian(AGN_data_W1.iloc[:, 1].tolist())}')
    print(f'AGN - Median number of measurements in epoch (W2) = {np.nanmedian(AGN_data_W2.iloc[:, 1].tolist())}')
    print(f'AGN - Median number of measurements in an epoch when mean unc used (W1) = {np.nanmedian(AGN_epoch_measurements_mean_unc_W1)}')
    print(f'AGN - Median number of measurements in an epoch when median unc used (W1) = {np.nanmedian(AGN_epoch_measurements_median_unc_W1)}')
    print(f'AGN - Median number of measurements in an epoch when mean unc used (W2) = {np.nanmedian(AGN_epoch_measurements_mean_unc_W2)}')
    print(f'AGN - Median number of measurements in an epoch when median unc used (W2) = {np.nanmedian(AGN_epoch_measurements_median_unc_W2)}')


# creating a 2d plot of W1 NFD vs W1 number of epochs
if epochs_NFD_W1 == 1:
    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_W1_epochs, AGN_W1_NFD, color='blue',  label='Non-CL AGN')
    plt.scatter(CLAGN_W1_epochs, CLAGN_W1_NFD, color='red',  label='CLAGN')
    plt.xlim(0, 1.05*max(AGN_W1_epochs+CLAGN_W1_epochs))
    plt.ylim(0, 1.05*max(AGN_W1_NFD+CLAGN_W1_NFD))
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.xlabel("W1 Epochs", fontsize = 24)
    plt.ylabel("W1 NFD", fontsize = 24)
    plt.title("W1 NFD vs W1 Epochs", fontsize = 24)
    plt.legend(loc = 'best', fontsize=22)
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
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.xlabel("W2 Epochs", fontsize = 24)
    plt.ylabel("W2 NFD", fontsize = 24)
    plt.title("W2 NFD vs W2 Epochs", fontsize = 24)
    plt.legend(loc = 'best', fontsize=22)
    plt.tight_layout()
    plt.show()


# creating a 2d plot of W1 Z-Score vs W1 number of epochs
if epochs_zs_W1 == 1:
    plt.figure(figsize=(12, 7))
    plt.scatter(AGN_W1_epochs, AGN_W1_zscore_mean, color='blue',  label='Non-CL AGN')
    plt.scatter(CLAGN_W1_epochs, CLAGN_W1_zscore_mean, color='red',  label='CLAGN')
    plt.xlim(0, 1.05*max(AGN_W1_epochs+CLAGN_W1_epochs))
    plt.ylim(0, 1.05*max(AGN_W1_zscore_mean+CLAGN_W1_zscore_mean))
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.xlabel("W1 Epochs", fontsize = 24)
    plt.ylabel("W1 Z-Score", fontsize = 24)
    plt.title("W1 Z-Score vs W1 Epochs", fontsize = 24)
    plt.legend(loc = 'best', fontsize=22)
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
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.xlabel("W2 Epochs", fontsize = 24)
    plt.ylabel("W2 Z-Score", fontsize = 24)
    plt.title("W2 Z-Score vs W2 Epochs", fontsize = 24)
    plt.legend(loc = 'best', fontsize=22)
    plt.tight_layout()
    plt.show()


if redshift_dist_CLAGN_vs_non_CLAGN == 1:
    combined_redshifts = AGN_redshifts_analysis+CLAGN_redshifts_analysis
    redshift_binsize = (max(combined_redshifts)-min(combined_redshifts))/20 #20 bins
    bins_redshift = np.arange(min(combined_redshifts), max(combined_redshifts) + redshift_binsize, redshift_binsize)

    fig, ax1 = plt.subplots(figsize=(12,7))
    ax2 = ax1.twinx()
    hist1 = ax1.hist(AGN_redshifts_analysis, bins=bins_redshift, color='blue', edgecolor='black', alpha=0.6)
    hist2 = ax2.hist(CLAGN_redshifts_analysis, bins=bins_redshift, color='red', edgecolor='black', alpha=0.6)
    line1 = ax1.axvline(median_AGN_redshift_analysis, linewidth=2, linestyle='-', color='black')
    line2 = ax2.axvline(median_CLAGN_redshift_analysis, linewidth=2, linestyle=':', color='black')
    ax1.tick_params(axis='x', labelsize=22)
    ax1.tick_params(axis='y', labelsize=22)
    ax2.tick_params(axis='y', labelsize=22)
    ax1.set_xlabel('Redshift', fontsize = 22)
    ax1.set_ylabel('Non-CL AGN Frequency', color='blue', fontsize = 22)
    ax2.set_ylabel('CLAGN Frequency', color='red', fontsize = 22)
    handles = [hist1[2][0], hist2[2][0], line1, line2]
    labels = ['Non-CL AGN Control', 'Guo CLAGN', f'Non-CL AGN Median = {median_AGN_redshift_analysis:.2f}', f'CLAGN Median = {median_CLAGN_redshift_analysis:.2f}']
    ax1.legend(handles, labels, loc='upper right', fontsize = 21)
    if brightness == 0:
        plt.title('Redshift Distribution - Dim CLAGN & Non-CL AGN Analysed', fontsize=24)
    elif brightness == 1:
        plt.title('Redshift Distribution - Bright CLAGN & Non-CL AGN Analysed', fontsize=24)
    elif brightness == 2:
        plt.title('Redshift Distribution - CLAGN & Non-CL AGN Analysed', fontsize=24)    
    plt.tight_layout()
    plt.show()


if redshift_dist_bright_dim == 1:
    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[np.where(CLAGN_quantifying_change_data.iloc[:, 27].notna(),  
        CLAGN_quantifying_change_data.iloc[:, 27] >= bright_dim_W1,  
        CLAGN_quantifying_change_data.iloc[:, 30] >= bright_dim_W2)]
    CLAGN_names_analysis_bright = CLAGN_quantifying_change_data.iloc[:, 0].tolist()
    CLAGN_redshifts_bright = []
    for object_name in CLAGN_names_analysis_bright:
        object_row = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
        redshift = object_row.iloc[0, 3]
        CLAGN_redshifts_bright.append(redshift)

    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[np.where(CLAGN_quantifying_change_data.iloc[:, 27].notna(),  
        CLAGN_quantifying_change_data.iloc[:, 27] < bright_dim_W1,  
        CLAGN_quantifying_change_data.iloc[:, 30] < bright_dim_W2)]
    CLAGN_names_analysis_dim = CLAGN_quantifying_change_data.iloc[:, 0].tolist()
    CLAGN_redshifts_dim = []
    for object_name in CLAGN_names_analysis_dim:
        object_row = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
        redshift = object_row.iloc[0, 3]
        CLAGN_redshifts_dim.append(redshift)

    combined_redshifts = CLAGN_redshifts_bright+CLAGN_redshifts_dim
    redshift_binsize = (max(combined_redshifts)-min(combined_redshifts))/20 #20 bins
    bins_redshift = np.arange(min(combined_redshifts), max(combined_redshifts) + redshift_binsize, redshift_binsize)
    AGN_median_redshift = np.median(AGN_redshifts)
    CLAGN_median_redshift_bright = np.median(CLAGN_redshifts_bright)
    CLAGN_median_redshift_dim = np.median(CLAGN_redshifts_dim)
    plt.figure(figsize=(12,7))
    plt.hist(CLAGN_redshifts_bright, bins=bins_redshift, color='black', histtype='step', linewidth=2, label='Bright CLAGN')
    plt.hist(CLAGN_redshifts_dim, bins=bins_redshift, color='gray', alpha=0.7, label='Dim CLAGN')
    plt.axvline(CLAGN_median_redshift_bright, linewidth=2, linestyle='-', color='darkred', label = f'Bright CLAGN Median = {CLAGN_median_redshift_bright:.2f}')
    plt.axvline(CLAGN_median_redshift_dim, linewidth=2, linestyle='--', color='darkred', label = f'Dim CLAGN Median = {CLAGN_median_redshift_dim:.2f}')
    plt.xlabel('Redshift')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Redshifts - Dim vs Bright CLAGN')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if luminosity_dist_CLAGN == 1:
    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[np.where(CLAGN_quantifying_change_data.iloc[:, 27].notna(),  
        CLAGN_quantifying_change_data.iloc[:, 27] >= bright_dim_W1,  
        CLAGN_quantifying_change_data.iloc[:, 30] >= bright_dim_W2)]
    CLAGN_names_analysis_bright = CLAGN_quantifying_change_data.iloc[:, 0].tolist()
    CLAGN_W1_low_flux = CLAGN_quantifying_change_data.iloc[:, 27].tolist()

    CLAGN_luminosity_brightflux = []
    for object_name, min_flux in zip(CLAGN_names_analysis_bright, CLAGN_W1_low_flux):
        if np.isnan(min_flux):
            continue
        else:
            object_row = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
            redshift = object_row.iloc[0, 3]
            CLAGN_luminosity_brightflux.append(luminosity(min_flux, redshift))

    CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_max_uncs.csv')
    CLAGN_quantifying_change_data = CLAGN_quantifying_change_data[np.where(CLAGN_quantifying_change_data.iloc[:, 27].notna(),  
        CLAGN_quantifying_change_data.iloc[:, 27] < bright_dim_W1,  
        CLAGN_quantifying_change_data.iloc[:, 30] < bright_dim_W2)]
    CLAGN_names_analysis_dim = CLAGN_quantifying_change_data.iloc[:, 0].tolist()
    CLAGN_W1_low_flux = CLAGN_quantifying_change_data.iloc[:, 27].tolist()

    CLAGN_luminosity_dimflux = []
    for object_name, min_flux in zip(CLAGN_names_analysis_dim, CLAGN_W1_low_flux):
        if np.isnan(min_flux):
            continue
        else:
            object_row = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
            redshift = object_row.iloc[0, 3]
            CLAGN_luminosity_dimflux.append(luminosity(min_flux, redshift))

    combined_luminosities = [lum.to_value(u.erg / (u.s * u.AA)) for lum in CLAGN_luminosity_brightflux + CLAGN_luminosity_dimflux]
    combined_lambda_lum = [lum*(3.4*10**-6) for lum in combined_luminosities]
    CLAGN_luminosity_brightflux = [lum*(3.4*10**-6) for lum in CLAGN_luminosity_brightflux]
    CLAGN_luminosity_dimflux = [lum*(3.4*10**-6) for lum in CLAGN_luminosity_dimflux]
    bins_mod_dev = np.logspace(np.log10(min(combined_lambda_lum)), np.log10(max(combined_lambda_lum)), num=20)
    CLAGN_median_luminosity_bright = np.median([lum.to_value(u.erg / (u.s * u.AA)) for lum in CLAGN_luminosity_brightflux])
    CLAGN_median_luminosity_dim = np.median([lum.to_value(u.erg / (u.s * u.AA)) for lum in CLAGN_luminosity_dimflux])
    plt.figure(figsize=(12,7))
    plt.hist(CLAGN_luminosity_brightflux, bins=bins_mod_dev, color='black', histtype='step', linewidth=2, label='Bright CLAGN')
    plt.hist(CLAGN_luminosity_dimflux, bins=bins_mod_dev, color='gray', alpha=0.7, label='Dim CLAGN')
    plt.axvline(CLAGN_median_luminosity_bright, linewidth=2, linestyle='-', color='darkred', label = f'Bright Flux CLAGN Median = {CLAGN_median_luminosity_bright:.2e}')
    plt.axvline(CLAGN_median_luminosity_dim, linewidth=2, linestyle='--', color='darkred', label = f'Dim Flux CLAGN Median = {CLAGN_median_luminosity_dim:.2e}')
    plt.xscale('log')
    plt.xlabel(r'λ$L_{λ}$ / erg s$^{-1}$', fontsize=26)
    plt.ylabel('Frequency', fontsize=26)
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.title(f'Distribution of Luminosities - Dim vs Bright CLAGN', fontsize=28)
    plt.legend(loc='best', fontsize=15)
    plt.tight_layout()
    plt.show()


if luminosity_dist_AGN == 1:
    AGN_quantifying_change_data = pd.read_csv(f'AGN_Quantifying_Change_just_MIR_max_uncs_Sample_{my_sample}.csv')
    AGN_quantifying_change_data = AGN_quantifying_change_data[np.where(AGN_quantifying_change_data.iloc[:, 27].notna(),  
        AGN_quantifying_change_data.iloc[:, 27] >= bright_dim_W1,  
        AGN_quantifying_change_data.iloc[:, 30] >= bright_dim_W2)]
    AGN_names_analysis_bright = AGN_quantifying_change_data.iloc[:, 0].tolist()
    AGN_W1_low_flux = AGN_quantifying_change_data.iloc[:, 31].tolist()

    AGN_luminosity_brightflux = []
    for object_name, min_flux in zip(AGN_names_analysis_bright, AGN_W1_low_flux):
        if np.isnan(min_flux):
            continue
        else:
            object_row = AGN_sample[AGN_sample.iloc[:, 3] == object_name]
            redshift = object_row.iloc[0, 2]
            AGN_luminosity_brightflux.append(luminosity(min_flux, redshift))

    AGN_quantifying_change_data = pd.read_csv(f'AGN_Quantifying_Change_just_MIR_max_uncs_Sample_{my_sample}.csv')
    AGN_quantifying_change_data = AGN_quantifying_change_data[np.where(AGN_quantifying_change_data.iloc[:, 27].notna(),  
        AGN_quantifying_change_data.iloc[:, 27] < bright_dim_W1,  
        AGN_quantifying_change_data.iloc[:, 30] < bright_dim_W2)]
    AGN_names_analysis_dim = AGN_quantifying_change_data.iloc[:, 0].tolist()
    AGN_W1_low_flux = AGN_quantifying_change_data.iloc[:, 31].tolist()

    AGN_luminosity_dimflux = []
    for object_name, min_flux in zip(AGN_names_analysis_dim, AGN_W1_low_flux):
        if np.isnan(min_flux):
            continue
        else:
            object_row = AGN_sample[AGN_sample.iloc[:, 3] == object_name]
            redshift = object_row.iloc[0, 2]
            AGN_luminosity_dimflux.append(luminosity(min_flux, redshift))

    combined_luminosities = [lum.to_value(u.erg / (u.s * u.AA)) for lum in AGN_luminosity_brightflux + AGN_luminosity_dimflux]
    combined_lambda_lum = [lum*(3.4*10**-6) for lum in combined_luminosities]
    AGN_luminosity_brightflux = [lum*(3.4*10**-6) for lum in AGN_luminosity_brightflux]
    AGN_luminosity_dimflux = [lum*(3.4*10**-6) for lum in AGN_luminosity_dimflux]
    bins_mod_dev = np.logspace(np.log10(min(combined_lambda_lum)), np.log10(max(combined_lambda_lum)), num=20)
    AGN_median_luminosity_bright = np.median([lum.to_value(u.erg / (u.s * u.AA)) for lum in AGN_luminosity_brightflux])
    AGN_median_luminosity_dim = np.median([lum.to_value(u.erg / (u.s * u.AA)) for lum in AGN_luminosity_dimflux])
    plt.figure(figsize=(12,7))
    plt.hist(AGN_luminosity_brightflux, bins=bins_mod_dev, color='black', histtype='step', linewidth=2, label='Bright Non-CL AGN')
    plt.hist(AGN_luminosity_dimflux, bins=bins_mod_dev, color='gray', alpha=0.7, label='Dim Non-CL AGN')
    plt.axvline(AGN_median_luminosity_bright, linewidth=2, linestyle='-', color='darkred', label = f'Bright Flux Non-CL AGN Median = {AGN_median_luminosity_bright:.2e}')
    plt.axvline(AGN_median_luminosity_dim, linewidth=2, linestyle='--', color='darkred', label = f'Dim Flux Non-CL AGN Median = {AGN_median_luminosity_dim:.2e}')
    plt.xscale('log')
    plt.xlabel(r'λ$L_{λ}$ / erg s$^{-1}$', fontsize=26)
    plt.ylabel('Frequency', fontsize=26)
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.title(f'Distribution of Luminosities - Dim vs Bright Non-CL AGN', fontsize=28)
    plt.legend(loc='upper left', fontsize=15)
    plt.tight_layout()
    plt.show()


if UV_NFD_redshift == 1:
    CLAGN_UV_all = pd.read_csv('CLAGN_Quantifying_Change_UV_all.csv')
    CLAGN_UV_all = CLAGN_UV_all.dropna(subset=[CLAGN_UV_all.columns[1]])
    combined_CLAGN_UV_NFD = CLAGN_UV_all.iloc[:, 1]
    combined_CLAGN_redshift = CLAGN_UV_all.iloc[:, 5]

    CLAGN_UV_all_Halpha = CLAGN_UV_all[CLAGN_UV_all.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'Halpha'].iloc[:, 0])]
    CLAGN_UV_NFD_Halpha = CLAGN_UV_all_Halpha.iloc[:, 1]
    CLAGN_redshift_Halpha = CLAGN_UV_all_Halpha.iloc[:, 5]
    print(f'number of Halpha CLAGN = {len(CLAGN_UV_NFD_Halpha)}')

    CLAGN_UV_all_Hbeta = CLAGN_UV_all[CLAGN_UV_all.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'Hbeta'].iloc[:, 0])]
    CLAGN_UV_NFD_Hbeta = CLAGN_UV_all_Hbeta.iloc[:, 1]
    CLAGN_redshift_Hbeta = CLAGN_UV_all_Hbeta.iloc[:, 5]
    print(f'number of Hbeta CLAGN = {len(CLAGN_UV_NFD_Hbeta)}')

    CLAGN_UV_all_Mg2 = CLAGN_UV_all[CLAGN_UV_all.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'Mg ii'].iloc[:, 0])]
    CLAGN_UV_NFD_Mg2 = CLAGN_UV_all_Mg2.iloc[:, 1]
    CLAGN_redshift_Mg2 = CLAGN_UV_all_Mg2.iloc[:, 5]
    print(f'number of Mg2 CLAGN = {len(CLAGN_UV_NFD_Mg2)}')

    CLAGN_UV_all_C3_ = CLAGN_UV_all[CLAGN_UV_all.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'C iii]'].iloc[:, 0])]
    CLAGN_UV_NFD_C3_ = CLAGN_UV_all_C3_.iloc[:, 1]
    CLAGN_redshift_C3_ = CLAGN_UV_all_C3_.iloc[:, 5]
    print(f'number of C3_ CLAGN = {len(CLAGN_UV_NFD_C3_)}')

    CLAGN_UV_all_C4 = CLAGN_UV_all[CLAGN_UV_all.iloc[:, 0].isin(Guo_table4_filled[Guo_table4_filled['Line'] == 'C iv'].iloc[:, 0])]
    CLAGN_UV_NFD_C4 = CLAGN_UV_all_C4.iloc[:, 1]
    CLAGN_redshift_C4 = CLAGN_UV_all_C4.iloc[:, 5]
    print(f'number of C4 CLAGN = {len(CLAGN_UV_NFD_C4)}')
    
    plt.figure(figsize=(12, 7))
    plt.scatter(CLAGN_redshift_Halpha, CLAGN_UV_NFD_Halpha, s=100, color='rosybrown',  label=u'H\u03B1 CLAGN')
    plt.scatter(CLAGN_redshift_Hbeta, CLAGN_UV_NFD_Hbeta, s=100, color='brown',  label=u'H\u03B2 CLAGN')
    plt.scatter(CLAGN_redshift_Mg2, CLAGN_UV_NFD_Mg2, s=100, color='red',  label=u'Mg ii CLAGN')
    plt.scatter(CLAGN_redshift_C3_, CLAGN_UV_NFD_C3_, s=100, color='salmon',  label=u'C iii] CLAGN')
    plt.axhline(y=three_sigma_UV_NFD, color='black', linestyle='--', linewidth=2, label = 'Threshold')
    plt.xlim(0, 1.05*max(combined_CLAGN_redshift))
    plt.ylim(0, 1.05*max(combined_CLAGN_UV_NFD))
    plt.tick_params(axis='both', labelsize=26, length=8, width=2)
    plt.xlabel("UV NFD", fontsize = 26)
    plt.ylabel("Frequency", fontsize = 26)
    plt.title("Comparing UV Variability for different CLAGN BELs", fontsize = 28)
    plt.legend(loc = 'best', fontsize=25)
    plt.tight_layout()
    plt.show()