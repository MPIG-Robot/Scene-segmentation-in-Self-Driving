# 无人驾驶中的场景分割

该代码实现了基于无人驾驶下的道路场景分割功能。  
## 网络
我们使用了全卷积神经网络(FCN)来训练模型  
## 实验环境
ubuntu16.04 ，Anaconda3  
数据集是剑桥大学camvid数据集  
Pytorch框架进行训练
## 参考
主要参考了：https://github.com/meetshah1995/pytorch-semseg
## 特点
相对于参考代码，使用了更适合道路分割的camvid数据集，编写了配置文件 fcn8s_camvid.yml文件，用以训练自己的模型
### 联系邮箱
ifanxi1998@gmail.com
