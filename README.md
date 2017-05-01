# ewc_chainer_cnn_mnist

## Description
Experiment with learning at elastic weight consolidation (EWC) using chainer.  
Learning data: mnist.  
Learning model: cnn.  

Original paper: Overcoming catastrophic forgetting in neural networks  
https://arxiv.org/abs/1612.00796  

Reference implementation  
https://github.com/ariseff/overcoming-catastrophic

## Requirement
python 2.7+  
chainer 2.0.0a1  

## Usage
First task: normal classification task (acc_tr is acuuracy of learning data, acc_te is accuracy of test data)  
`$python train1_ewc_cnn.py`  

Second task: a task of classifying learning data and test data,using fliped image. transfer learning from the first task. (Acc_tr is acuuracy of learning data, acc_te is accuracy of test data, acc_an_t is accuraxy of test data of the first task)  
`$python train2_ewc_cnn.py`  

Third task: Learn the second task using ewc_loss.  
`$python train2_ewc_cnn_F.py` 
