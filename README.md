# 3D-Few-Shot-Semantic-Segmentation
3D Few-Shot Semantic Segmentation

## Installation
- Install `python` --This repo is tested with `python 3.10.8`.
- Install `pytorch` with CUDA -- This repo is tested with `torch 1.13.1`, `CUDA 11.7`. 
It may work with newer versions, but that is not gauranteed.
- Install `faiss` with cpu version
	```
	pip install faiss-cpu
	```
- Install 'torch-cluster' with the corrreponding torch and cuda version
	```
	pip install torch-cluster
	```
- Install dependencies
    ```
    pip install tensorboard h5py transforms3d matplotlib
    ```

## Usage
### Data preparation
#### S3DIS
1. Download [S3DIS Dataset Version 1.2](http://buildingparser.stanford.edu/dataset.html).
2. Re-organize raw data into `npy` files by running
   ```
   cd ./preprocess
   python collect_s3dis_data.py --data_path $path_to_S3DIS_raw_data
   ```
   The generated numpy files are stored in `./datasets/S3DIS/scenes/data` by default.
3. To split rooms into blocks, run 

    ```python ./preprocess/room2blocks.py --data_path ./datasets/S3DIS/scenes/```
    
    One folder named `blocks_bs1_s1` will be generated under `./datasets/S3DIS/` by default. 


#### ScanNet
1. Download [ScanNet V2](http://www.scan-net.org/).
2. Re-organize raw data into `npy` files by running
	```
	cd ./preprocess
	python collect_scannet_data.py --data_path $path_to_ScanNet_raw_data
	```
   The generated numpy files are stored in `./datasets/ScanNet/scenes/data` by default.
3. To split rooms into blocks, run 

    ```python ./preprocess/room2blocks.py --data_path ./datasets/ScanNet/scenes/ --dataset scannet```
    
    One folder named `blocks_bs1_s1` will be generated under `./datasets/ScanNet/` by default. 


### Running 
#### Training
First, pretrain the segmentor which includes feature extractor module on the available training set:
    
    bash ./scripts/pretrain_segmentor.sh

Second, train our method:
	
	bash ./scripts/train_attMPTI.sh


#### Evaluation
    
    bash ./scripts/eval_attMPTI.sh

Note that the above scripts are used for 2-way 1-shot on S3DIS (S^0). You can modified the corresponding hyperparameters to conduct experiments on other settings. 



## Citation
Please cite our paper if it is helpful to your research:

    @inproceedings{zhao2021few,
      title={Few-shot 3D Point Cloud Semantic Segmentation},
      author={Zhao, Na and Chua, Tat-Seng and Lee, Gim Hee},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2021}
    }


## Acknowledgement
We thank [DGCNN (pytorch)](https://github.com/WangYueFt/dgcnn/tree/master/pytorch) for sharing their source code.
