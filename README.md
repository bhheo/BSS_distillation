# Knowledge Distillation with Adversarial Samples Supporting Decision Boundary

Official Pytorch implementation of paper:

[Knowledge Distillation with Adversarial Samples Supporting Decision Boundary](https://arxiv.org/abs/1805.05532) (AAAI 2019).

Sporlight and poster are available on [homepage](https://sites.google.com/view/byeongho-heo/home)

## Environment
Python 3.6, Pytorch 0.4.1, Torchvision


## Knowledge distillation [(CIFAR-10)](https://www.cs.toronto.edu/~kriz/cifar.html) 

```shell
python train_BSS_distillation.py 
```


Distillation from ResNet 26 (teacher) to ResNet 10 (student) on CIFAR-10 dataset.

Pre-trained teacher network (ResNet 26) is included.


## Citation

```
@inproceedings{BSSdistill,
	title = {Knowledge Distillation with Adversarial Samples Supporting Decision Boundary},
	author = {Byeongho Heo, Minsik Lee, Sangdoo Yun, Jin Young Choi},
	booktitle = {AAAI},
	year = {2019}
}
```
