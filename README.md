# CXR_Covid-19

Labelling
```
index = {'normal': 0, 'covid': 1,  'pneumonia': 2}
rev_index = {0: 'normal',1: 'covid', 2: 'pneumonia'}
```

To copy from source to destination, go to gpuserver3
```
scp CXR_Covid-19_Challenge.zip sysadm@gpuserver2:/home/sysadm/sb/
scp CXR_Covid-19_Challenge.zip sysadm@gpuserver1:/home/sysadm/sb1/
```

To copy from destination (border machine) to home
```
scp  -P 4422 sysadm@ssh.rkmvu.ac.in:/home/sysadm/CXR_Covid-19_50e_InceptionResNetV2_360x360_128-32-3.h5 /home/
```

To copy from gpuserver3 to border machine, go to border machine and run this command 
```
scp  sysadm@gpuserver3:~/sb/CXR_Covid-19/Comparison/CXR_Covid-19_50e_InceptionResNetV2_360x360_128-32-3.h5 /home/sysadm/
```

ablation 100 epoch 
```
VGG_16 : gpu 3
Run for 100 epoch Validation Inception V3
Run for 100 epoch Full Inception V3
```

Comparison: 50 epochs

```
VGG_16 : done
MobileNetV2: done
NASNetLarge: done
InceptionResNetV2: done
ResNet152V2: done
Xception: done
DenseNet201: done
InceptionV3: done
EfficientNetB1: done
EfficientNetB7: done in gpu2
```
EfficientNets showing very erratic nature for small batch sizes.\
Best to leave them, considering their poor overall performance.

Best: 50 epoch with whole data - Training
```
InceptionResNetV2: done
InceptionV3: gpu3
```

# If you find this work useful, please consider citing

## Our paper

```

```

## For Dataset

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

