# CXR_Covid-19

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

Comparison: 50 epochs

```
VGG_16 : done
MobileNetV2: done
NASNetLarge: done
InceptionResNetV2: done
ResNet152V2: done
EfficientNetB7: gpu2 (with base)
```

Best: 50 epoch with whole data - Training
```
InceptionResNetV2: gpu-3
```
