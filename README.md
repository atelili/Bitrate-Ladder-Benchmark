# Bitrate-Ladder-Benchmark


Benchmarking Learning-based Bitrate Ladder Prediction Methods for Adaptive Video Streaming

<p align="center">
  <img src="https://github.com/Tlili-ahmed/2BiVQA/blob/master/figures/2BiVQA_overview2.drawio.png">
</p>


This repository contains the code for our paper on [Benchmarking Learning-based Bitrate Ladder Prediction Methods for Adaptive Video Streaming](#benchmarking_learning-based_bitrate_ladder_prediction_methods_for_adaptive_video_streaming). 
If you use any of our code, please cite:
```
@article{Telili2022,
  title = {Benchmarking Learning-based Bitrate Ladder Prediction Methods for Adaptive Video Streaming},
  author = {Ahmed Telili, Wassim Hamidouche, Sid Ahmed Fezza, and Luce Morin},
  booktitle={IEEE TRANSACTIONS ON IMAGE PROCESSING},
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
  * [References](#references)
    
<!-- /code_chunk_output -->



## Requirements
```python
pip install -r requirements.txt
```

## Features extraction

#### a-Handcrafted features:

```
python features_extration  [-h] [-r 'path to raw videos directory']
                                   [-f 'path to meta-data csv file']
                                   [-o 'overlapping between patches']
```
#### b-Deep features:

python features_extration  [-h] [-v 'path to raw videos directory']
                                   [-f 'path to meta-data csv file']
                                   [-np 'number of patches']
                                   [-nf 'number of frames']
                                   [-m 'backbone model']
                                   [-o 'overlapping between patches']

Please note that we provide four pretrained backbone models for features extraction: resnet50, densenet169, vgg16 and inception_v3.




## Model Training :

#### a-Handcrafted features:


Training can be started by importing Bitrate_Ladder.ipynb in Google Colab or Jupyter Notebook.

#### b-Deep features:

```python
python train.py  [-h] [-v 'path to raw videos directory']
                                   [-np 'number of patches']
                                   [-nf 'number of frames']
                                   [-b 'batch_size (1)']```




## Performance Benchmark:



  
| Methods \ Scores |R2          | SROCC            | PLCC        | ACCURACY | BD-BR vs GT | BD-BR vs AL | BD-BR vs RL |
|:------------:|:-----------:|:---------------:|:-----------:|:--------:|:-----------:|:-----------:|:-----------:|
| ExtraTrees Regressor      | 0.7635 / 0.6420   | 0.8174 / 0.6635       | 0.9000 / 0.8277  | 0.8779 / 0.8400    |  1.433% / 2.704% | -18.427% / -18.827% |-9.025% / -8.798%


## References


```
[1] V. Hosu, F. Hahn, M. Jenadeleh, H. Lin, H. Men, T. Szirányi, S. Li,and D. Saupe, “The konstanz natural video database (konvid-1k),” in2017 Ninth international conference on quality of multimedia experience(QoMEX).  IEEE, 2017, pp. 1–6.

[2] Z. Sinno and A. C. Bovik, “Large-scale study of perceptual videoquality,”IEEE Transactions on Image Processing, vol. 28, no. 2, pp.612–627, 2018.

[3] Y. Wang, S. Inguva, and B. Adsumilli, “Youtube ugc dataset for videocompression research,” in2019 IEEE 21st International Workshop onMultimedia Signal Processing (MMSP).  IEEE, 2019, pp. 1–5.
```




