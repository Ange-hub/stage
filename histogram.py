import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Data_SHP.csv', delimiter=';')

grain_area = df.area[df['DN'] == 253]
list(np.float_(grain_area))

"""
Crée un histogramme de la distribution de la taille des grains geodataframe
Pré-requis : le fichier Data_SHP.csv qui contient le geodataframe des polygones (voir  png_to_shp.py)

"""


# plt.hist(np.log10(grain_area), 50)
# plt.yscale('log')
# plt.title('Distribution de la taille des grains')
# plt.ylabel('Occurences')
# plt.xlabel('Aire des grains en log10 (m²)')
# plt.show()

# print(np.mean(grain_area))
# print(np.min(grain_area))

ratio = []

file_list = list(df.file.astype(str).str.cat(df.num.astype(str)))

for file_name in np.unique(file_list[:]):
    sum_grain = 0
    sum_bg = 0
    idx = np.where(np.array(file_list)==file_name)[0]
    if len(idx) > 1:
        for index in idx:
            if df['DN'][index] == 253:
                sum_grain += df.area[index]
            else:
                sum_bg += df.area[index]
        ratio.append(sum_grain/(sum_grain+sum_bg))

print(np.mean(ratio)*100)
print(np.max(ratio)*100)
print(np.min(ratio)*100)

plt.hist(np.array(ratio)*100, 100)
plt.legend(loc='upper right')
plt.yscale('log')
plt.title('Distribution de la proportion des grains')
plt.ylabel('Occurences')
plt.xlabel('Proportion (%)')
plt.show()

