# Influencer-Backdoor-Attack-on-Semantic-Segmentation

This repository is the Python implementation of the paper **"Influencer Backdoor Attack on Semantic Segmentation"** presented at **ICLR 2024**.



## Project Setup

#### Requirements

- Python version: **3.8**
- **Software:** PyTorch >= 1.1.0, Python 3, tensorboardX, apex

#### Clone the Repository

```bash
git clone https://github.com/Maxproto/Influencer-Backdoor-Attack-on-Semantic-Segmentation.git
```

#### Dataset Setup

Download related datasets and symlink the paths to them as follows (you can alternatively modify the relevant paths specified in the folder config):

```bash
cd semseg
mkdir -p dataset
ln -s /path_to_cityscapes_dataset dataset/cityscapes
```

#### Pre-trained Models

Download ImageNet pre-trained models from [this link](https://drive.google.com/open?id=15wx9vOM0euyizq-M1uINgN0_wjVRf9J3) and place them under the `initmodel` folder for weight initialization.



## Running the Project

#### 1. Modify the Attack Configuration

Specify the attack configuration in the config

- **train_NNI & train_PRL**: Whether to use Nearest Neighbor Injection or Pixel Random Labeling in the training process (default value: False)
- **test_NNI & test_PRL**: Whether to use Nearest Neighbor Injection or Pixel Random Labeling in the test process (default value: True)
- **train_scale & test_scale**: The rescale ratio of the images and labels (default cityscapes dataset: 0.5; VOC2012 dataset: 1)
- **trigger_size**: The pixel size of the injected trigger pattern (default cityscapes dataset: 55; VOC2012 dataset: 15)
- **lower_dist & upper_dist**: The distance range of the injected trigger and the victim class object
- **num_poison**: Number of poison images
- **num_perturb**: Number of pixel modified in the PRL process

#### 2. Training and Testing

Specify the GPU in the config then perform training and testing:

```bash
sh tool/train.sh cityscapes deeplabv3
sh tool/test.sh cityscapes deeplabv3
```



## Applying Attack on Other Models

You could use the poisoned images and labels generated in the attack process to train other models, using the same pipeline of your models. The poisoned images and labels can be found in the dataset root under the folder with names starting with `poisonedImage_` & `poisonedFolder_`.



## Citation

If you find this project useful for your research, please consider citing:

```BibTeX
@inproceedings{lan2024influencer,
    title={Influencer Backdoor Attack on Semantic Segmentation},
    author={Haoheng Lan and Jindong Gu and Philip Torr and Hengshuang Zhao},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=VmGRoNDQgJ}
}
```



## Acknowledgments

This source code is inspired by the codebase [semseg](https://github.com/hszhao/semseg).



## Contact Information

For any inquiries or further information regarding this project, feel free to reach out to the author **Max Haoheng Lan**, haohenglan@outlook.edu.
