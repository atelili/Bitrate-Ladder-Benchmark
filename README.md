# Bitrate-Ladder-Benchmark


Benchmarking Learning-based Bitrate Ladder Prediction Methods for Adaptive Video Streaming

<p align="center">
  <img src="https://github.com/atelili/Bitrate-Ladder-Benchmark/blob/master/Figures/overview.png">
</p>

This repository contains the code for our paper on [Benchmarking Learning-based Bitrate Ladder Prediction Methods for Adaptive Video Streaming](#benchmarking_learning-based_bitrate_ladder_prediction_methods_for_adaptive_video_streaming). 
If you use any of our code, please cite:
```
@article{Telili2022,
  title = {Benchmarking Learning-based Bitrate Ladder Prediction Methods for Adaptive Video Streaming},
  author = {Ahmed Telili, Wassim Hamidouche, Sid Ahmed Fezza, and Luce Morin},
  year = {2022}
}
```



  * [Requirements](#requirements)
  * [Features extraction](#features-extraction)
      * [Handcrafted features]
      * [Deep features]
  * [Model Training](#model-training)
      * [Handcrafted features-based models]
      * [Deep features-based models]

  * [Performance Benchmark](#performance-benchmark)
 
    
<!-- /code_chunk_output -->



## Requirements
```python
pip install -r requirements.txt
```

## Features extraction

#### a- Handcrafted features:

```
python features_extration  [-h] [-r 'path to raw videos directory']
                                   [-f 'path to meta-data csv file']
                                   [-o 'overlapping between patches']
```
#### b- Deep features:

python features_extration  [-h] [-v 'path to raw videos directory']
                                   [-f 'path to meta-data csv file']
                                   [-np 'number of patches']
                                   [-nf 'number of frames']
                                   [-m 'backbone model']
                                   [-o 'overlapping between patches']

Please note that we provide four pretrained backbone models for features extraction: resnet50, densenet169, vgg16 and inception_v3.




## Model Training :

#### a- Handcrafted features:


Training can be started by importing Bitrate_Ladder.ipynb in Google Colab or Jupyter Notebook.

#### b- Deep features:

```python
python train.py  [-h] [-v 'path to raw videos directory']
                                   [-np 'number of patches']
                                   [-nf 'number of frames']
                                   [-b 'batch_size (1)']

```


## Performance Benchmark:


#### a-YPSNR quality metric:

  
| Methods \ Scores |R2          | SROCC            | PLCC        | ACCURACY | BD-BR vs GT | BD-BR vs AL | BD-BR vs RL |
|:------------:|:-----------:|:---------------:|:-----------:|:--------:|:-----------:|:-----------:|:-----------:|
|ExtraTrees Regressor| 0.7635    | 0.8174        | 0.9000   | 0.8779     |  1.433%  | -18.427%  |-9.025% 
|XGBoost      | 0.6165 | 0.7560 | 0.8278 | 0.8578 | 2.320% | -18.099% | -8.706%|
|Gaussian Process| 0.6390 | 0.7620 | 0.8473 | 0.8566 | 1.740% | -18.244% | -6.286% |
|Random Forest Regressor| 0.6758 | 0.7993 | 0.8440 | 0.8671 | 1.535% | -18.324% | -8.879% |
|Densenet169| 0.4725 | 0.6423 | 0.7756 | 0.8166 | 3.380% | -15.669%  | -8.169% |
|VGG16| 0.5172 | 0.5236 | 0.7652 | 0.8223 | 3.083% | -15.536% | -8.088% |
|ResNet-50| 0.4564 | 0.5680 | 0.7457 | 0.8483 | 2.424%| -15.806% | -8.300% |
|EfficientNet B7| 0.4237 | 0.5649 | 0.7159 | 0.8004 | 3.396% | -15.506% | -8.012% |

#### b-VMAF quality metric:
| Methods \ Scores |R2          | SROCC            | PLCC        | ACCURACY | BD-BR vs GT | BD-BR vs AL | BD-BR vs RL |
|:------------:|:-----------:|:---------------:|:-----------:|:--------:|:-----------:|:-----------:|:-----------:|
|ExtraTrees Regressor| 0.6420 | 0.6635 | 0.8277 | 0.8400 | 2.704% | -18.827% | -8.798%|
|XGBoost| 0.5533 | 0.6470 | 0.7997 | 0.8347 | 3.444% | -18.650% | -8.608%|
|Gaussian Process| 0.4292 | 0.4918 | 0.6983 | 0.8012 | 5.254% | -18.328% | -7.688%|
|Random Forest Regressor| 0.5899 | 0.6564 | 0.8059 | 0.8300 | 3.052% | -18.887% | -8.616%|
|Densenet169| 0.4216 | 0.6167 | 0.6433 | 0.7901 | 3.820% | -15.892% | -7.851%|
|VGG16| 0.4992 | 0.5112 | 0.7601 | 0.8052 | 4.125% | -15.812% | -7.593%|
|ResNet-50| 0.4045 | 0.5367 | 0.6962 | 0.8278 | 2.969% | -15.941% | -7.810%|
|EfficientNet B7| 0.3920 | 0.5612 | 0.6905 | 0.7781 | 4.742% | -15.771% | -7.607%|




