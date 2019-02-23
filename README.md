# Single-Image Depth Estimation Based on Fourier Domain Analysis

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
<br />
A PyTorch implementation of the [article](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2873.pdf), presented at CVPR 2018.
## Abstract
This work considers the well-known problem of single image depth estimation. We implement the Depth Estimation Network (DEN), Depth-Balanced Euclidean (DBE) loss and the Fourier Domain Combination (FDC) model of the original paper in PyTorch. At the time of writing this poster, it had provided state-of-the-art performance. Since then, few papers outperformed the proposed algorithm slightly but it is still one of the top algorithms in the comparison tables.

## Architecture

### Deep Estimation Network
Base on ResNet-152, where its last 19 ResNet blocks are modified
to extract intermediate features. All the extracted intermediate features are concatenated and fed into a fully connected layer to obtain the estimated depth map.
![Alt text](https://github.com/shalomma/DEN/blob/master/.github/den.png)

### Depth-Balanced Euclidean Loss
A modification of the commonly used Euclidean loss, which helps to more relaibly estimate shallow depths, as well as deep depths.
![Alt text](https://github.com/shalomma/DEN/blob/master/.github/dbe.png)

### Fourier Domain Combination
Depth map candidates are generated using the Deep Estimation Network. By cropping each image in different ratios we get a batch of candidates, Transform the candidates to the frequency with the 2D DFT, and finally, Linearly combine the frequency maps to a single one. We then apply the inverse 2D DFT to obtain the final estimated depth map.
![Alt text](https://github.com/shalomma/DEN/blob/master/.github/fdc.png)

## Results
![Alt text](https://github.com/shalomma/DEN/blob/master/.github/results.png)

## Development
### Prerequisites
* Python 3
* PyTorch 1.0.0
* TorchVision 0.2.1
<br />
You can install all the requirements in a virtual environment as follows

```sh
$ python -m pip install --user denenv
$ pip install -r requirements.txt
```

### Data Preparation
For a quick implementation, download the slim [NYU V2](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) dataset (.mat file), place it in the data directory and run the converter script to parse the file into RGB images and depth maps.

```sh
$ mkdir data/nyu_v2/
$ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
$ mv nyu_depth_v2_labeled.mat ./data/nyu_v2/
$ python converter.py
```
### Training

```sh
$ run.py
```


### Inference
Implement in your code as follows
```
import torch
from den import DEN
...
den = DEN()
den(img)
```


## Citation
```
@inproceedings{
  author = {Jae-Han Lee and Minhyeok Heo and Kyung-Rae Kim and Shih-En Wei and Chang-Su Kim},
  booktitle = {CVPR},
  title = {Single-Image Depth Estimation Based on Fourier Domain Analysis},
  year = {2018}
}
```
