# ECE 228 Group 27
### Preperation
 1. Download TuSimple dataset from https://www.kaggle.com/datasets/manideep1108/tusimple to the parent directory of this repo.
 2. Copy `../TUSimple/test_label.json` to `../TUSimple/test_set/test_label.json`. This will allow the labels to be loaded in testing stage.
### Training
To reproduce the results in the report, you can use the following commands:

```
python train.py --prefix segnet --model segnet
python train.py --prefix enet --model enet
python train.py --prefix enet_weight_0.1 --model enet --weight 0.1
python train.py --prefix enet_k --model enet_k
python train.py --prefix enet_k_weight_0.1 --model enet_k --weight 0.1
python train.py --prefix resnet38 --model resnet38
python train.py --prefix resnet38_weight_0.1 --model resnet38 --weight 0.1
python train.py --prefix resnet38_k --model resnet38_k
python train.py --prefix resnet38_k_weight_0.1 --model resnet38_k --weight 0.1
```

### Evaluation
The `inference.ipynb` file is used to check the performance of trained models. The `eval_loss.ipynb` is used to calculate the evaluation loss, including the soft evaluation loss at testing stage.

### References
https://github.com/Wizaron/instance-segmentation-pytorch

https://github.com/jaeoh2/Road-Lane-Instance-Segmentation-PyTorch

