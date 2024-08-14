# MGFIN
This is the implementation code of the paper “Multi-Granularity Feature Interaction Network for Zero-Shot Learning”.


## Requirements
- Python 3.8.16
- PyTorch = 1.8.0
- Torchvision = 0.9.0
- numpy = 1.23.5

## Training
- **Dataset**: please download the dataset, (i.e., CUB, SUN and AWA2) to the dataset root path on your machine
- **Data split**: Datasets can be download from [Xian et al. (CVPR2017)](https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip) and take them into dir ```../../datasets/```.
- Download pre-trained vision Transformer as the vision backbone.

### Training Script
```shell
python train_m.py --config-file config/cub_model.yaml
python train_m.py --config-file config/sun_model.yaml
python train_m.py --config-file config/awa2_model.yaml
```

## Testing
<!-- You can download the pre-trained model on three different datasets: CUB, SUN, AWA2 in the CZSL/GZSL setting.  -->
### Training Script
```shell
python test.py --config-file config/cub_model.yaml
python test.py --config-file config/sun_model.yaml
python test.py --config-file config/awa2_model.yaml
```
