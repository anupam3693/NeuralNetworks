##  Summary of Experiment carried out for CIFAR 10 Data using Densenet Architecture

**Top Two Experiment Results**:

Below are the **hyperparameters and parameters** value mentioned for two experiment 

(Exp1_Run1 & Exp1_Run2) :

​	Training Accuracy achieved : **98.68% (0.9868)**

​	Test Accuracy achieved : **93.20% (0.9320)**

​	Total Number of epochs: **74**

(Exp2_Run1 & Exp2_Run2) :

​	Training Accuracy achieved : **99.32% (0.9932)**

​	Test Accuracy achieved : **92.58% (0.9258)**

​	Total Number of epochs: **144**

| Versions             | Exp1_Run1                                                    | Exp1_Run2                                                    | Exp2_Run1                                             | Exp2_Run2                                              |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------ |
| Batch size           | 64                                                           | 64                                                           | 64                                                    | 64                                                     |
| Layer                | 20                                                           | 20                                                           | 14                                                    | 14                                                     |
| Filter               | 28                                                           | 28                                                           | 20                                                    | 20                                                     |
| Compression          | 0.9                                                          | 0.9                                                          | 0.9                                                   | 0.9                                                    |
| Dropout              | 0.2                                                          | 0.2                                                          | 0.2                                                   | 0.2                                                    |
| Total params         | 875,028                                                      | 875,028                                                      | 965,074                                               | 965,074                                                |
| Trainable params     | 868,462                                                      | 868,462                                                      | 952,018                                               | 952,018                                                |
| Non-trainable params | 6,566                                                        | 6,566                                                        | 13,056                                                | 13,056                                                 |
| Epochs               | 63                                                           | 11                                                           | 93                                                    | 51                                                     |
| Val_acc              | 0.9579                                                       | 0.9786                                                       | 0.9130                                                | 0.9258                                                 |
| Train_acc            | 0.9189                                                       | 0.9343                                                       | 0.9559                                                | 0.9932                                                 |
| Momentum             | 0.9                                                          | 0.9                                                          | 0.9                                                   | 0.9                                                    |
| Decay                | 10e-4                                                        | 10e-4                                                        | 10e-4                                                 | 10e-4                                                  |
| Steps per epoch      | 1562                                                         | 50000                                                        | 1562                                                  | 50000                                                  |
| Image.Aug Used       | Yes                                                          | No                                                           | Yes                                                   | No                                                     |
| rotation_range       | 15                                                           | -                                                            | 15                                                    | -                                                      |
| width_shift_range    | 0.1                                                          | -                                                            | 0.1                                                   | -                                                      |
| height_shift_range   | 0.1                                                          | -                                                            | 0.1                                                   | -                                                      |
| zoom_range           | 0.1                                                          | -                                                            | 0.1                                                   | -                                                      |
| horizontal_flip      | True                                                         | -                                                            | True                                                  | -                                                      |
| LR Method            | LearningRateScheduler                                        | LearningRateScheduler                                        | CyclicLR                                              | CyclicLR                                               |
|                      | lrate = 0.1<br/>    if epoch > 40:
        lrate = 0.01
    elif epoch > 75:
        lrate = 0.001 
    elif epoch > 100:
        lrate = 0.0001 | lrate = 0.1<br/>    if epoch > 40:
        lrate = 0.01
    elif epoch > 75:
        lrate = 0.001 
    elif epoch > 100:
        lrate = 0.0001 | base_lr=0.1
max_lr=0.3
step_size=2000
scale_mode='cycle' | base_lr=0.01
max_lr=0.1
step_size=2000
scale_mode='cycle' |
| Load weight          | No                                                           | Yes (from s1_v1)                                             | No                                                    | Yes (from s2_v1)                                       |
| Time Taken in hrs    | 3.73                                                         | 0.79                                                         | approx 5                                              | 1.18                                                   |
|                      |                                                              |                                                              |                                                       |                                                        |
|                      |                                                              |                                                              |                                                       |                                                        |

Links for experiment 

Exp1_Run1 :

​	https://github.com/anupam3693/eip2/blob/master/Anupam_Kumar_EIP2_Batch2_Assignment_DNST_CIFAR10_AUG_exp1_run1.ipynb

 Exp1_Run2 ( weight loaded from Exp1_Run1 ) :

​	https://github.com/anupam3693/eip2/blob/master/Anupam_Kumar_EIP2_Batch2_Assignment_DNST_CIFAR10_AUG_exp1_run2.ipynb



Exp1_Run1:

​	https://github.com/anupam3693/eip2/blob/master/Anupam_Kumar_EIP2_Batch2_Assignment_DNST_CIFAR10_AUG_exp2_run1.ipynb

Exp1_Run2 ( weight loaded from Exp1_Run1 ) :

​	https://github.com/anupam3693/eip2/blob/master/Anupam_Kumar_EIP2_Batch2_Assignment_DNST_CIFAR10_AUG_exp2_run2.ipynb



**Experiment 1 Summary:**

Major changes and tweaking done in **Run1**:

1. Inclusion of Con2D_1x1 in Dense block

   Introducing this layer with 28  filters to reduce the feature maps size and the perform a more
   expensive 3x3 convolution.

   ```
   Dense_Conv2D_1_1 = Conv2D(num_filter, (1,1), use_bias=False ,padding='same')(temp)
   ```

2. Elimination of last block and passing Third Transition to the Output Layer

   ```
   Third_Transition = add_transition(Third_Block, num_filter, dropout_rate)
   
   # Last_Block = add_denseblock(Third_Transition,  num_filter, dropout_rate=0.2)
   output = output_layer(Third_Transition)
   ```

3. Hyper Parameter considered:

   ```
   batch_size = 64
   num_classes =  10
   epochs = 200
   l = 20
   num_filter = 28
   compression = 0.9
   dropout_rate = 0.2
   ```

4. Learning Rate considered - LearningRateScheduler

   Smaller step sizes results in the NN learning a more exact solution hence the overfitting. A moderate learning rate would overshoot such points never settling but oscillating about such a point hence likely to generalize better than smaller steps.

   ```
       lrate = 0.1
       if epoch > 40:
           lrate = 0.01
       elif epoch > 75:
           lrate = 0.001 
       elif epoch > 100:
           lrate = 0.0001   
       return lrate
   ```

5. Image Augmentation: 

   Image augmentation is required to boost the performance of deep neural networks. **Image augmentation** helps in creating training images through different ways of processing or combination such as random rotation, shifts, shear and flips, etc.

Major changes and tweaking done in  **Run2** :

1. No Image Augmentation used

   Removing image augmentation here helps in getting a peak in validation accuracy probably.May be after removing image augmentation model learned in better way on original dataset after learning from augmented images.

2. Learning Rate changed made :

   Learning rate was continued with the last value used in previous run subsequently using 0.001 which finals helps in achieving maximum accuracy.

   ```
       lrate = 0.01
       if epoch > 20:
           lrate = 0.001
       elif epoch > 50:
           lrate = 0.0001 
       return lrate
   ```



**Experiment 2 Summary:**

Though validation accuracy was more in experiment 1 but the training accuracy is more in Experiment 2.Probably model is overfitting more here o using cyclicLR

Major changes and tweaking done in **Run1**:

Same as Experiment 1 Run 1 except below:

1. Hyper Parameter considered:

1. ```
   batch_size = 64
   num_classes =  10
   epochs = 200
   l = 14
   num_filter = 20
   compression = 0.9
   dropout_rate = 0.2
   ```

2.  Learning methodology used is CyclicLR.

   ```
   clr = CyclicLR(base_lr=0.1, max_lr=0.3,
                                   step_size=2000., scale_fn=clr_fn,
                                   scale_mode='cycle')
   ```



Major changes and tweaking done in **Run2**:

1. No Image Augmentation used

2. Parameters changes made in CyclicLR

```
clr = CyclicLR(base_lr=0.01, max_lr=0.1,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
```



# Understanding of the Learning Rate:

The higher the learning rate more the models learns with every epoch hence lesser epoch is required to learn more information and converge better.  It has been observed that with high learning rate the accuracy gets stuck at some value hence remain stagnent(around .80's- 0.90's). The work around is to have the lower learning rate put into the place so that the model can come out of the local minima also referred as saddle point. Every model and dataset has got different learning rate hence need to closely watch the epoch vs train/test accuracy  as well as train/test loss to inncorporate right learning rate. 
This is trial based as nothing concrete has been defined as what learning rate should be selected. This is purely experimantal based selection. 

Tried with below learning rate methodology:
1. SGDR
2. keras LearningRateScheduler with step decay
3. keras ReduceLROnPlateau
4. Cyclic Learning Rate
5. Manually by varying SGD parameter lr
6. LR Finder

I have used LearningRateScheduler in my submitted model though tested all and not major difference I found excelt the fact that LRscheduler was easy to modify and implement and you could clearly recognize the effect.

# Batch size:

It has been observed that the lesser batch size(64) means slow learning but the accuracy seems to be more. With high batch size(128) learning is faster but the accuracy is slightly lesser.

# Densenet Architecure changes
Removed the last block from the given architecture observed the learning was fast as well accuracy was better. I couldn't understand why it is like that.

Also added the 1x1 convolution in the dense layer which improved the learning drastically. This is because with this layer model tends to learn more without increase in the parameters hence resulting into more accuracy.

# Epochs

With more epochs the validation as well as train accuracy gets stabalize.

Some keras classes helped to achieve the target really well and constrained the whole learning by monitering the learning growth. Some classes used were : callback option with EarlyStopping, CheckPointer.



