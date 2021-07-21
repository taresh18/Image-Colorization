# Colorizing Balck & White Images Using Generative Adverserial Networks

Implemeted a Generational Adverserial Network for colorizing black & white images. <br />
Dataset used : [COCO-2017 training](https://www.kaggle.com/awsaf49/coco-2017-dataset)  <br />
Original paper : [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)

<img src="https://user-images.githubusercontent.com/58368119/126567319-569ba7a9-3e2a-4bf0-a417-0b23f17001be.png" width="90%"></img>
<br />
<br />

<img src="https://user-images.githubusercontent.com/58368119/126567435-61b85cd1-6a5d-4a79-93cd-02466e5f9269.png" width="90%"></img>







## Things I Tried:

* Label Smoothing : Put the two target labels, Real = 0.9 and Fake = 0.1, instead of 1 and 0 respectively.
* Pre-trained the Generator separately:
  * Used U-Net with a pretrained (on ImageNet) Resnet model as its backbone and the pretrained the whole generator model on the dataset.
  * Tried with Resnet18 and Resnet34 models.
* Tried different Image Augmentations:
  both train-time (from Albumentation library) and test-time (from TTA library)
* Tried different optimisers for Generator and Discriminator.

## Things to try in Future:

* BatchNorm : Construct different mini-batches for real and fake images.
* Training for more epochs.
* Hyper-parameter optimisations
* Create a web application and deploy the model over it.


