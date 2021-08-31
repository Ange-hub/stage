#!Users\Ange\anaconda3\envs\gdal_env python

import geopandas as gpd
import pandas as pd
from pathlib import Path
import os

"""
Convertit des images .png en .shp et crée un geodataframe contenant toutes les géométries
Pré-requis : un environnement contenant GDAL
"""

def png_to_shp(png_dir, shp_dir):
    
    png_dir = Path(png_dir)
    shp_dir = Path(shp_dir)

    shp_list = []

    #supp les fichiers existants dans le shp_dir
    for file in shp_dir.iterdir():
        file.unlink()

    wld_parameters = ['2e-6', '0.0000000000', '0.0000000000', '-2e-6', '1e-6', '511e-6']

    for i, png_path in enumerate(png_dir.glob('*.png')):
        
        png_name = png_path.stem

        wld_path = png_dir.joinpath(f'{png_name}.wld')

        with open(wld_path, 'w') as file:
            file.writelines(l + '\n' for l in wld_parameters)

        #faire les conversions avec GDAL
        tif_path = shp_dir.joinpath(f'{png_name}.tif')
        shp_path = shp_dir.joinpath(f'{png_name}.shp')

        cmd = 'gdal_translate -of GTiff -q ' + str(f'{png_path} {tif_path}')
        os.system(cmd)

        cmd = 'gdal_polygonize.py -q ' + str(f'{tif_path} {shp_path}')
        os.system(cmd)

        #ouvrir le shp dans un geodataframe temporaire
        temp = gpd.read_file(shp_path)

        #colonnes en plus
        temp['area'] = temp.geometry.area  # le perimetre de chaque polygone sur l'image 
        temp['perimeter'] = temp.geometry.length  # l'aire de chaque polygone sur l'image 
        temp['file'] = png_name.split('_')[0]
        temp['num'] = png_name.split('_')[1]
        
        #ajouter le geodataframe à shp_list
        shp_list.append(temp)

        #supprimer le fichier WLD créé
        wld_path.unlink()

    #fin de l'iteration, on sauvegarde les dataframe
    gdf = gpd.GeoDataFrame(pd.concat(shp_list))
    gdf.index.names = ['index']
    gdf.to_csv('Data_SHP.csv', sep=';')
    return gdf


png_dir = r'C:\Users\Ange\Desktop\preds_output'
shp_dir = r'C:\Users\Ange\Desktop\shp_ouput'
png_to_shp(png_dir, shp_dir)