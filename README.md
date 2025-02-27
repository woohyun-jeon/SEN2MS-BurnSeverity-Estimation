# Burn severity mapping with Sentinel-2 images
This repository contains the implementation of burn severity mapping with bi-temporal Sentinel-2 multi-spectral images

## Prerequisites
* python >= 3.8
* torch >= 2.1.0
* torchvision >= 0.16.0

## Usage
1) Clone the repository and install the required dependencies with the following command:
```
$ git clone https://github.com/woohyun-jeon/SEN2MS-BurnSeverity-Estimation.git
$ cd SEN2MS-BurnSeverity-Estimation
$ pip install -r requirements.txt
```
2) Download datasets from here: https://drive.google.com/drive/folders/10jHd-mJX3e5rOBeLzjTteEKiYIENK28K?usp=drive_link

The directory structure should be as follows:
```
  image/
    after/  
        0000.tif
        0001.tif
        ...
    before/
        0000.tif
        0001.tif
        ...    
  label/
    0000.tif
    0001.tif
    ... 
  train.txt
  valid.txt
  test.txt
```
* It is important to mention that "data_path" argument in "configs.yaml" file, denoting the parent directory of image & label path, should be properly adjusted.
* Plus, "out_path" argument, indicating output directory of prediction and model files, should be properly adjusted.

3) Run main.py code with the following command:
```
$ cd src
$ python main.py
```