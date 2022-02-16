# Chest X-Ray COVID-19 Detection
***

## Work presented at the [Ethics and Explainability for Responsible Data Science (EE-RDS) Conference 2021](https://www.uj.ac.za/event/ethics-and-explainability-for-responsible-data-science-ee-rds/).
***


## Download related models and datasets

https://drive.google.com/drive/u/2/folders/1Pf9FyzaiTqF3HkJ6nR_UHtefVKGc5gcZ


Note: (While using the model, use these labels)

```
index = {'normal': 0, 'covid': 1,  'pneumonia': 2}
rev_index = {0: 'normal',1: 'covid', 2: 'pneumonia'}
```

## Some relevant stuffs from paper

<center>
  <img src="https://raw.githubusercontent.com/Jimut123/CXR_Covid-19/main/assets/ablation.png">
  <img src="https://raw.githubusercontent.com/Jimut123/CXR_Covid-19/main/assets/tables.png">
  <img src="https://raw.githubusercontent.com/Jimut123/CXR_Covid-19/main/assets/vizs.png">
</center>

## Acknowledgements

The authors are thankful to **Moulay Akhloufi** for sharing the datasets.

The authors are also grateful to **Swathy Prabhu Mj**, Ramakrishna Mission Vivekananda Educational and Research Institute, for arranging a machine with an Asus RTX 2080 Ti (12 GB VRAM) and 64 GB RAM, to hasten the research.

## If you find this work useful, please consider citing

### Our paper

```
@INPROCEEDINGS{9708580,  
  author={Pal, Jimut Bahan and Paul, Nilayan},  
  booktitle={2021 Ethics and Explainability for Responsible Data Science (EE-RDS)},   
  title={Classifying Chest X-Ray COVID-19 images via Transfer Learning},   
  year={2021},  
  volume={},  
  number={},  
  pages={1-8},  
  doi={10.1109/EE-RDS53766.2021.9708580}
  }
```

### For Dataset

```
@misc{CovidGrandChallenge2021,
  author = {Akhloufi, Moulay A. and Chetoui, Mohamed},
  title = {{Chest XR COVID-19 detection}},  
  howpublished = {\url{https://cxr-covid19.grand-challenge.org/}},
  month = {August},
  year = {2021},
  note = {Online; accessed September 2021},
  }
```

