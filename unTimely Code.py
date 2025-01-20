import pandas as pd
from unTimely_Catalog_tools import unTimelyCatalogExplorer

# object_name = '125449.57+574805.3' #Object R - chosen because not a CLAGN, but has a spurious measurement
object_name = '121449.54+572734.1' #Object Y - chosen because of enourmous NFD (nearly 7)

parent_sample = pd.read_csv('clean_parent_sample_no_CLAGN.csv')
Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")
object_data = parent_sample[parent_sample.iloc[:, 3] == object_name]

if len(object_data) == 0: #If a CLAGN; CLAGN are not in parent sample
    parent_sample = pd.read_csv('guo23_parent_sample_no_duplicates.csv')
    object_data = parent_sample[parent_sample.iloc[:, 3] == object_name]
    SDSS_RA = object_data.iloc[0, 0]
    SDSS_DEC = object_data.iloc[0, 1]
else:
    SDSS_RA = object_data.iloc[0, 0]
    SDSS_DEC = object_data.iloc[0, 1]

ucx = unTimelyCatalogExplorer(directory='C:/Users/ciara/Dropbox/University/University Work/Fourth Year/Project/unTimely Charts', cache=True, show_progress=True, timeout=300, suppress_console_output=False,
                              catalog_base_url='https://unwise.me/data/neo7/untimely-catalog/',
                              catalog_index_file='untimely_index-neo7.fits')

result_table = ucx.search_by_coordinates(SDSS_RA, SDSS_DEC, box_size=100, cone_radius=None, show_result_table_in_browser=False,
                                         save_result_table=True, result_table_format='ascii.ipac', result_table_extension='dat')

ucx.create_finder_charts(overlays=True, overlay_color='green', overlay_labels=False, overlay_label_color='red',
                     image_contrast=3, open_file=False, file_format='pdf')

ucx.create_image_blinks(blink_duration=300, image_zoom=10, image_contrast=5, separate_scan_dir=False, display_blinks=False)