# Overview   #概述   概述
We construct the dataset using a case-control scenario sampling scheme. To address the challenge of estimating the actual class prior probability, we propose using the VPU method for pretraining. This approach effectively enhances model performance and ensures stable training of the classification model, even when the class prior is unknown. To mitigate the issue of imbalanced distribution of positive samples, we introduce a curriculum learning strategy. This allows the model to first learn from simple samples and gradually adapt to more complex data distributions, simulating the human learning process from easy to difficult and improving the model's generalization ability.我们使用病例控制场景抽样方案构建数据集。为了解决估计实际类先验概率的挑战，我们提出使用VPU方法进行预训练。这种方法有效地提高了模型的性能，保证了分类模型训练的稳定性，即使在类先验未知的情况下。为了缓解正样本分布不平衡的问题，我们引入了课程学习策略。这使得模型可以先从简单的样本中学习，逐渐适应更复杂的数据分布，模拟人类由易到难的学习过程，提高模型的泛化能力。
![1743150655570](https://github.com/user-attachments/assets/342caad3-25e6-4d43-b2da-8998f67827ea)

## Preparing Data   ##准备数据
Download Alzheimers and OASIS via url, i.e. https://www.kaggle.com/datasets/ninadaithal/imagesoasis and https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset下载阿尔茨海默氏症和绿洲通过网址，即https://www.kaggle.com/datasetsDownload Alzheimers and OASIS via url, i.e. https://www.kaggle.com/datasets
inadaithal/imagesoasis and https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset下载阿尔茨海默氏症和绿洲通过网址，即https://www.kaggle.com/datasets
inadaithal/imagesoasis和https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset


## Requirements   # #要求
* Python 3
* Pytorch >= 1.0
* Scikit-learn >= 0.2
* Numpy >=1.1   *麻木>=1.1
* transformers >=4.49.0   *变压器>=4.49.0

## Details   # #细节
If not pre-trained, set arg.pretrained to None. And then the weight of the training is given to arg.pretrained.如果没有预先训练，设置参数。预训练为零。然后将训练的权重赋给arg。pretrained。
## Reference   # #参考
[1] Masahiro Kato and Takeshi Teshima and Junya Honda. "Learning from Positive and Unlabeled Data with a Selection Bias." International Conference on Learning Representations. 2019.加藤正弘、铁岛武和本田纯弥。“从带有选择偏差的正面和未标记数据中学习。”学习表征国际会议。2019。

[2] Chen H, Liu F, Wang Y, et al. A variational approach for learning from positive and unlabeled data[J]. Advances in Neural Information Processing Systems, 2020, 33: 14844-14854.[10]陈海，刘峰，王勇，等。基于变分方法的非标记数据学习方法[J]。神经信息处理系统进展，2020,33(3):14844-14854。

[3] Zhu Z, Wang L, Zhao P, et al. Robust positive-unlabeled learning via noise negative sample self-correction[C]//Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2023: 3663-3673.[10]朱忠，王磊，赵鹏，等。基于噪声负样本自校正的鲁棒性正无标签学习[j] .中文信息学报，2016,36(1):559 - 563。
## Acknowledgments   致谢
My implementation has been inspired from the following sources.我的实现受到以下来源的启发。

* [Robust](https://github.com/woriazzc/robust-pu) : I have mainly followed the Pytorch Version of this Repo. We understood the probability of curriculum learning through this code.
* [Grad-CAM](https://github.com/zhanghailan123/SVM_visualization/tree/main) - We accomplished the drawing of the image through this code.
* [VPU](https://github.com/HC-Feynman/vpu) - I have followed this repository to incorporate loss in my implementation.
