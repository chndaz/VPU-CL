# Overview  
We construct the dataset using a case-control scenario sampling scheme. To address the challenge of estimating the actual class prior probability, we propose using the VPU method for pretraining. This approach effectively enhances model performance and ensures stable training of the classification model, even when the class prior is unknown. To mitigate the issue of imbalanced distribution of positive samples, we introduce a curriculum learning strategy. This allows the model to first learn from simple samples and gradually adapt to more complex data distributions, simulating the human learning process from easy to difficult and improving the model's generalization ability.
![1743762032842](https://github.com/user-attachments/assets/89b2745c-3bd2-468f-be06-bcd481c65486)


## Preparing Data  
Download Alzheimers and OASIS via url, i.e. https://www.kaggle.com/datasets/ninadaithal/imagesoasis and https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset




## Requirements   
* Python 3
* Pytorch >= 1.0
* Scikit-learn >= 0.2
* Numpy >=1.1   
* transformers >=4.49.0   

## Details   
If not pre-trained, set arg.pretrained to None. And then the weight of the training is given to arg.pretrained.
## Reference   
[1] Masahiro Kato and Takeshi Teshima and Junya Honda. "Learning from Positive and Unlabeled Data with a Selection Bias." International Conference on Learning Representations. 2019.
[2] Chen H, Liu F, Wang Y, et al. A variational approach for learning from positive and unlabeled data[J]. Advances in Neural Information Processing Systems, 2020, 33: 14844-14854.

[3] Zhu Z, Wang L, Zhao P, et al. Robust positive-unlabeled learning via noise negative sample self-correction[C]//Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2023: 3663-3673.

[4] Chen X, Chen W, Chen T, et al. Self-pu: Self boosted and calibrated positive-unlabeled training[C]//International Conference on Machine Learning. PMLR, 2020: 1510-1519.

## Acknowledgments   
My implementation has been inspired from the following sources.

* [Robust](https://github.com/woriazzc/robust-pu) : I have mainly followed the Pytorch Version of this Repo. We understood the probability of curriculum learning through this code.
* [Grad-CAM](https://github.com/zhanghailan123/SVM_visualization/tree/main) - We accomplished the drawing of the image through this code.
* [VPU](https://github.com/HC-Feynman/vpu) - I have followed this repository to incorporate loss in my implementation.
* [random forest](https://github.com/yuchaozhi/image-segmentation)- We completed some comparative experiments by referring to it.
  
