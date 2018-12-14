# Knowledge Distillation with Adversarial Samples Supporting Decision Boundary

Official Pytorch implementation of paper:

[Knowledge Distillation with Adversarial Samples Supporting Decision Boundary](https://arxiv.org/abs/1805.05532) (AAAI 2019).


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
@inproceedings{ABdistill,
	title = {Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons},
	author = {Byeongho Heo, Minsik Lee, Sangdoo Yun, Jin Young Choi},
	booktitle = {AAAI},
	year = {2019}
}
```

Byeongho Heo, Minsik Lee, Sangdoo Yun, Jin Young Choi, "Knowledge Distillation with Adversarial Samples Supporting Decision Boundary", CoRR, 2018. (AAAI at Feb. 2019)



