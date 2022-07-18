# Chest X-Ray COVID-19 Detection
***

## Work presented at the [Ethics and Explainability for Responsible Data Science (EE-RDS) Conference 2021](https://www.uj.ac.za/event/ethics-and-explainability-for-responsible-data-science-ee-rds/).
***

[J. B. Pal and N. Paul, "Classifying Chest X-Ray COVID-19 images via Transfer Learning," 2021 Ethics and Explainability for Responsible Data Science (EE-RDS), 2021, pp. 1-8, doi: 10.1109/EE-RDS53766.2021.9708580.](https://ieeexplore.ieee.org/document/9708580)

#### [[ Video recording of presentation](https://www.youtube.com/watch?v=27ixHn6SP_4) ] [ [Slides](https://jimut123.github.io/publications/CXR_19/CXR_COVID_19_slide.pdf) ]

## Abstract


The internal behavior of Deep Neural Network architectures can be difficult to interpret. Certain architectures achieve impressive feats in a particular dataset while failing to show comparable performance in other datasets. Developing an architecture that performs well on a dataset can be a time-consuming affair and computationally intensive process. This study explains the effect of transfer learning by fine-tuning already available state-of-the-art architectures in different datasets and using them to classify Chest X-Ray images with high accuracy. Using transfer learning helps the model learn problem-specific features in a short period. It further shows that different models perform differently in a particular setting for a dataset. Ablation studies show that a combination of smaller structures that gives an overall better result may not give the best result in the combined model. In addition, the “belief” of the model for selecting a particular class is visualized in this study.


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

