
![images.jpg](https://github.com/anupam3693/eip2/blob/master/images.jpg?raw=true)

## Convolution

A Convolution is a process of extracting one or more features comprising of edges, curves, color etc from an image using kernels (Filters). A filter is slid or convolve over an image at intervals called stride , until the whole image is captured. More features are divided into multiple channels and similar features is bagged into one channel. Mostly it picks up the bold or obvious things from the image depending upon how many number of convolution operation has been performed. 

High level convolution involves four major components:

1. Local Receptive Fields
2. Channels
3. Filters/kernels
4. Activation Functions and Pooling

## Filters/Kernels

A filter/kernel is known as feature extractor or feature indentifier. It is used to extract one or more features comprising of edges, curves, color etc of an image. Filters can be of any size but the standard size used in the Convolution industry is 3x3 and 1x1. 1X1 is considered to be best mixture. Filters can be of size 5x5, 7x7 and so on. It is basically an array of random numbers initiated called weights which is then multiplied and summed up with the image pixel area values being convoluted. This process will result into a single number which will be stored in single pixel in the next layer. 

## Epochs

An Epoch is one cycle when feature extraction is completed using lets say 3x3 filter on the given image. This process can be repeated many number of times, so if its repeated 2nd time then epoch is 2, 3 times then epoch is 3 and so on.  More number of epochs make sure that better feature capturing is done hence minimal amount of data loss which is not evident. This helps in proving good prediction as it uses same channels but with feature having more information.

## 1x1 convolution

1X1 Convolution is a process of extracting features from an image of size lets say HxWxD using 1x1 Filter. This helps is dimentional reductionality by selecting fewer kernel which would be suffient to pick the bold features from an image. Even this type of filters are best for creating mix images, still it serves the purpose. The resultant next layer will have same number of pixels as in the source. 

Consider 28x28 image with 20 features, if we apply 1x1x10 filter then the output will result into 28x28x10 channels. 

## 3x3 convolution

3X3 Convolution is a process of extracting features from an image of size lets say HxWxD using 3x3 Filter. This reduces the dimension of the source image by 2 every consecutive layer.

Consider 28x28 image with 20 features, if we apply 3x3x10 filter then the next layer output will result into 26x26x10 channels and so on. 

| Input    | Filter | Output   |
| -------- | ------ | -------- |
| 28x28x20 | 3x3x10 | 26x26x10 |

![seven.png](https://github.com/anupam3693/eip2/blob/master/seven.png?raw=true)


## Feature Maps

When the filter extract the image features, the resultant is a single pixel value. This actually represents the part of image of size of filter.  We call this as Feature maps. This is achieved when filter is initiated with random numbers called weights which is then multiplied and summed up with the image pixel area values being convoluted then this results into a single number which is stored in a single pixel in the next layer. 

## Feature Engineering (older computer vision concept)

Feature engineering is a process of identifying the right features to fit a model so that prediction can be achieved with high accuracy. This may lead to elmination of the existing features or addition of the  new feature till the required accuracy is met. This is widely practised in Applied Machine Learning area across domains.

## Activation Function

Activation Function is a function which helps us to supress the negative values and consider only positive values. RELU is one of the widely activation function being used in CNN which makes all negative values to zeros and consider only positive values. It is also an powerful way to handle the non-linear inputs to get mapped with the output. 

## How to create an account on Github and upload the sample project

1. Go to [www.github.com](http://www.github.com/)
2. Sign up by providing your Username, email and password field and click on Sign up button.
3. Choose "Unlimited public repositories for free" Plan.
4. You will receive a verification email from github to the inbox of email id provided.
5. click on the link provided in the email for confirmation.
6. Sign in to the github.com
7. Then you can update your profile details and set up email alerts at different level in "Notification" Tab.
8. You can upload your project and files in to github. First create a repository by clicking the upper-right corner of any page, click (+) , and then click New repository.
9. Enter your repository name ,choose public or private and click create repository button.
10. Copy the repository link

Then go to your command prompt, clone the project or github for desktop: Type following command in CMD console: git clone Create your project file under this directory.For example: git add sample.txt git commit -m "first commit" git push

To Verify:
Enter your git username and password. Now, you have created your project github.

## Receptive Fields

Receptive fileds are the area at the input image so that the specific features in terms of matrices whose size is equal to the given filter size can be  mapped to a single pixel in the next consecutive layer. Here too only bold features are picked up compared to weak ones. Example,

| Input   | Filter | Output |
| ------- | ------ | ------ |
| 28x28x1 | 3x3x1  | 26x26  |

## Some Examples of use of Mathjax in Markdown


$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$

$$ y=\sum_{i=1}^n g(x_i) $$

$$y = x^2 \hbox{ when $x > 2$}$$

$$\sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}$$

This package is basically useful in creating mathematical notation for different formulae in the .md file or on browser. 

