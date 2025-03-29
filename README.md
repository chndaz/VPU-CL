# Overview
We construct the dataset using a case-control scenario sampling scheme. To address the challenge of estimating the actual class prior probability, we propose using the VPU method for pretraining. This approach effectively enhances model performance and ensures stable training of the classification model, even when the class prior is unknown. To mitigate the issue of imbalanced distribution of positive samples, we introduce a curriculum learning strategy. This allows the model to first learn from simple samples and gradually adapt to more complex data distributions, simulating the human learning process from easy to difficult and improving the model's generalization ability.

## Preparing Data
Download Alzheimers and OASIS via url, i.e. https://www.kaggle.com/datasets/ninadaithal/imagesoasis and https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset
![1743150655570](https://github.com/user-attachments/assets/342caad3-25e6-4d43-b2da-8998f67827ea)
