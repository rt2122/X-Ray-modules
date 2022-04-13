import os
import pandas as pd
import shutil

def organize_data(inpath:str, outpath:str, tiles:list, 
        united_cat_path:str='~/Data/sim/filtered_cat_final.csv'):
    """
    At the start all data is located in one directory (inpath). This function creates separate 
    directory for each tile in (outpath).
    Filtered objects are transported from united_cat to each directory.
    """
    offset:int = min(tiles)

    all_obj = pd.read_csv(united_cat_path)
    for tile in tiles:
        set_num = tile - offset
        dirname = os.path.join(outpath, f'set_{set_num}')
        os.mkdir(dirname)
        for file in ['exp_tile{}.fits.gz', 'tile{}.fits.gz', 'tile{}.pkl', 'tile_bg{}.fits.gz']:
            shutil.copy(os.path.join(inpath, file.format('_' + str(tile))), 
                       os.path.join(dirname, file.format('')))
        
        #Get objects from this tile
        tile_obj = all_obj[all_obj['tile'] == tile] 
        tile_obj.to_csv(os.path.join(dirname, 'objects.csv'), index=False)
    
    return
