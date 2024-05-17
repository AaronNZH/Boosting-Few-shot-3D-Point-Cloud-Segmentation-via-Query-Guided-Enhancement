# Boosting Few-shot 3D Point Cloud Segmentation via Query-Guided Enhancement

<a href="https://arxiv.org/abs/2308.03177" target="_blank">[Arxiv]</a>

![teaser](framework.jpg)

## Installation
This repo is tested with `Ubuntu 18.04`
- Install `python` --This repo is tested with `python 3.10`.
- Install `pytorch` with CUDA -- This repo is tested with `torch 1.13.1`, `CUDA 11.7`. 
It may work with newer versions, but that is not guaranteed.
	```
	conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
	```
- Install dependencies
    ```
    pip install -r requirements.txt
    ```

## Usage
```
cd Boosting-Few-shot-3D-Point-Cloud-Segmentation-via-Query-Guided-Enhancement
```
### Data preparation
#### S3DIS
1. Download [S3DIS Dataset Version 1.2](http://buildingparser.stanford.edu/dataset.html).
2. Re-organize raw data into `npy` files by running
   ```
   python ./preprocess/collect_s3dis_data.py --data_path $path_to_S3DIS_raw_data
   ```
   The generated numpy files are stored in `./datasets/S3DIS/scenes/data` by default.
3. To split rooms into blocks, run 

    ```python ./preprocess/room2blocks.py --data_path ./datasets/S3DIS/```
    
    One folder named `blocks_bs1_s1` will be generated under `./datasets/S3DIS/` by default. 


#### ScanNet
1. Download [ScanNet V2](http://www.scan-net.org/).
2. Re-organize raw data into `npy` files by running
	```
	python ./preprocess/collect_scannet_data.py --data_path $path_to_ScanNet_raw_data
	```
   The generated numpy files are stored in `./datasets/ScanNet/scenes/data` by default.
3. To split rooms into blocks, run 

    ```python ./preprocess/room2blocks.py --data_path ./datasets/ScanNet/ --dataset scannet```
    
    One folder named `blocks_bs1_s1` will be generated under `./datasets/ScanNet/` by default. 


### Running 
#### Training
**We have prepared the pretrain and 1 way 1shot S<sup>0</sup> checkpoints in the log_s3dis and log_scannet folders.**

First, pretrain the segmentor which includes the feature extractor module on the available training set:
    
    bash ./scripts/pretrain_segmentor.sh

Second, train our method:
	
	bash ./scripts/train_attMPTI.sh


#### Evaluation
    
    bash ./scripts/eval_attMPTI.sh

Note that the above scripts are used under 1-way 1-shot S<sup>0</sup> setting on S3DIS. You can modify the corresponding hyperparameters to conduct experiments in other settings. 



## Citation
Please cite our paper if it is helpful to your research:
```
@inproceedings{10.1145/3581783.3612287,
author = {Ning, Zhenhua and Tian, Zhuotao and Lu, Guangming and Pei, Wenjie},
title = {Boosting Few-Shot 3D Point Cloud Segmentation via Query-Guided Enhancement},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
year = {2023},
doi = {10.1145/3581783.3612287},
}
```


## Acknowledgement
We thank [AttMPTI](https://github.com/Na-Z/attMPTI) for sharing their source code.
