import glob

# Use smote, data augmentation

"""
['COVID-19_Radiography_Dataset/Normal', 'COVID-19_Radiography_Dataset/Lung_Opacity', 'COVID-19_Radiography_Dataset/Viral Pneumonia', 'COVID-19_Radiography_Dataset/README.md.txt', 'COVID-19_Radiography_Dataset/Viral Pneumonia.metadata.xlsx', 'COVID-19_Radiography_Dataset/COVID.metadata.xlsx', 'COVID-19_Radiography_Dataset/Normal.metadata.xlsx', 'COVID-19_Radiography_Dataset/Lung_Opacity.metadata.xlsx', 'COVID-19_Radiography_Dataset/COVID']
"""

"""
covid  normal  pneumonia
"""

all_new_ds_normal = glob.glob('COVID-19_Radiography_Dataset/Normal/*')
print(len(all_new_ds_normal))
print(all_new_ds_normal[:10])

all_new_ds_covid = glob.glob('COVID-19_Radiography_Dataset/COVID/*')
print(len(all_new_ds_covid))
print(all_new_ds_covid[:10])


all_new_ds_vn = glob.glob('COVID-19_Radiography_Dataset/Viral Pneumonia/*')
print(len(all_new_ds_vn))
print(all_new_ds_vn[:10])





