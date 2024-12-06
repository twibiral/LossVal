# LossVal

Official implementation of [LossVal - Efficient Data Valuation for Neural Networks](https://arxiv.org/abs/2412.04158).

Data Valuation is the process if assigning an importance score to each data point in a dataset. 
This importance score can be used to improve the performance of a machine learning model by focusing on the most important data points or for better explaining your model. 
LossVal is a novel method for data valuation that is based on the idea of optimizing the importance scores as weights that are part of the loss function. LossVal is efficient, scalable, and can be used with any differentiable loss function.

In our experiments, we show that LossVal achieves state-of-the-art performance on a range of data valuation tasks, without needing any additional training run.

<div align='center'>

![f1scores](./figures/exp_1/f1_scores.png)

</div>


## Overview

In general, loss functions used with LossVal are of the form:

$$\text{LossVal} = \mathcal{L}\_{w}(y, \hat{y}) \cdot \text{OT}\_{w}(X\_{train}, X\_{val})^{2}$$

The model's prediction is denoted by $\hat{y}$, while $y$ represents the target values. 
The optimal transport distance $\text{OT}\_{w}$ takes the features of the training data $X\_{train}$ and validation data $X\_{val}$ as input. 
For the target loss $\mathcal{L}\_{w}$, we use instance-weighted formulations of existing loss functions, like a weighted cross-entropy loss or weighted mean-squared error loss (see below).

Weighted cross-entropy loss:

$$\text{CE}\_{w} = - \sum^{N}\_{n=1} \left[ w\_{n} \cdot \sum^{K}\_{k=1} y\_{n,k} \log(\hat y\_{n,k}) \right]$$

Weighted mean-squared error loss:

$$\text{MSE}\_{w} = \sum^{N}\_{n=1} w\_{n} \cdot (y\_{n} - \hat{y}\_{n})^2$$

Weighted optimal transport distance:

$$\text{OT}\_w(X\_{train}, X\_{val}) = \min\_{\gamma \in \Pi(w, 1)} \sum\_{n=1}^{N}\sum\_{j=1}^{J} c(x_n, x_j) \, \gamma\_{n,j}$$


## Use

You can find a basic reference implementation in [`src/lossval.py`](./src/LossVal.py).  Feel free to use this implementation as a starting point for your own experiments and modify to your needs.

All the data from the experiments can be found in the [`results`](./results) folder.


## Citation

If you use LossVal in your research, please cite our paper:

```
@misc{wibiral2024lossvalefficientdatavaluation,
      title={{L}oss{V}al: {E}fficient Data Valuation for Neural Networks}, 
      author={Tim Wibiral and Mohamed Karim Belaid and Maximilian Rabus and Ansgar Scherp},
      year={2024},
      eprint={2412.04158},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.04158}, 
}
```