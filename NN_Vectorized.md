# Understanding a Simple Neural Networks Learning for Multi-Class Classification -Maths Version

<img src="/Users/anupam7936/afb/Blogs/NN_Vectorized.assets/image-20200420215951820.png" alt="image-20200420215951820" style="zoom:50%;" />





We have learned through a simple Neural Networks for Binary Classification in a separate blogs. Now we will try to understand the working of Neural Networks for Multi-Class Classification. For the consistency of the paramters names across all layers we have used 'h' at the outer layer too. 



Summarizing Feed  Forward  parameters  for Multi-Class Classification:



## Hidden Layer 1

| Input/Output Parameters | Parameter Expressions                                        | Parameter Description                                        |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| $\large X_1$            | $\large\begin{bmatrix} x_{1} \\ x_{2} \\ . \\ x_n\\\end{bmatrix}\quad$ | Input vector (different features)                            |
| $\large W_1$            | $\large\begin{bmatrix} w_{111} & w_{112} & ... &w_{11n}  \\ w_{121} & w_{122} & ... &w_{12n} \\ ... & ... & ... &... \\ w_{1m1} & w_{1m} & ... &w_{1mn}\\ \end{bmatrix} \quad$ | $W_1$ matrix size is m x n, where, m = number of inputs and n = number of neurons |
| $\large b_1$            | $\large\begin{bmatrix} b_{11}  \\ b_{12}  \\ ...  \\ b_{1n} \\ \end{bmatrix} \quad$ | $b_1$ is a vector with size as n, number of neuron in layer 1 |
| $\large a_{1n}$         | $\large\begin{bmatrix} a_{11}  \\ a_{12}  \\ ...  \\ a_{1n} \\ \end{bmatrix} \quad = W_1 * X_1 + b_1$ | A pre-activation function vector of size n, number of neurons in layer 1. The Matrix multiplication size will be,  [n * m] * [m * 1 ]  + [n * 1]  = [n*1] |
| $\large h_{1n}$         | $\large \begin{bmatrix} h_{11}  \\ h_{12}  \\ ...  \\ h_{1n} \\ \end{bmatrix} \quad =\begin{bmatrix} g(a_{11})  \\ g(a_{12})  \\ ...  \\ g(a_{1n}) \\ \end{bmatrix} \quad$ | A activation function vector of size n, number of neurons in layer 1. The activation can be any non-linear continous function like Sigmoid, tanh etc. |

If the function to be considered is Sigmoid then, 



$\Large g(a_{11}) = \frac{1}{1+ e ^{-a_{11}}}$ and so on. 



### Hidden Layer 2



| Input/Output Parameters | Parameter Expressions                                        | Parameter Description                                        |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| $\large W_2$            | $\large\begin{bmatrix} w_{211} & w_{212} & ... &w_{21n}  \\ w_{221} & w_{222} & ... &w_{22n} \\ ... & ... & ... &... \\ w_{2m1} & w_{2m2} & ... &w_{2mn}\\ \end{bmatrix} \quad$ | $W_2$  matrix size is m x n, where, m = number of previous layer neurons  and n = number of neurons in the current layer |
| $\large b_2$            | $\large\begin{bmatrix} b_{21}  \\ b_{22}  \\ ...  \\ b_{2n} \\ \end{bmatrix} \quad$ | $b_2$ is a vector with size as n, number of neuron in layer 2 |
| $\large a_{2n}$         | $\large \begin{bmatrix} a_{21}  \\ a_{22}  \\ ...  \\ a_{2n} \\ \end{bmatrix} \quad = W_2 * h_1 + b_2 $ | A pre-activation function vector of size n, number of neurons in layer 2. The Matrix multiplication size will be [n * m ] * [m * 1] + [n*1] = [n * 1] |
| $\large h_{2n}$         | $\large \begin{bmatrix} h_{21}  \\ h_{22}  \\ ...  \\ h_{2n} \\ \end{bmatrix} \quad = \begin{bmatrix} g(a_{21})  \\ g(a_{22})  \\ ...  \\ g(a_{2n}) \\ \end{bmatrix} \quad$ | A activation function vector of size n, number of neurons in layer 2. The activation can be any non-linear continous function like Sigmoid, tanh etc. |

If the function to be considered is Sigmoid then, 



$\Large g^{'}(a_{21}) = \frac{1}{1+ e ^{-a_{21}}}$ and so on. 



## Output Layer

| Input/Output Parameters | Parameter Expressions                                        | Parameter Description                                        |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| $\large W_3$            | $\large\begin{bmatrix} w_{311} & w_{312} & ... &w_{31n}  \\ w_{321} & w_{322} & ... &w_{32n} \\ ... & ... & ... &... \\ w_{3m1} & w_{3m2} & ... &w_{3mn}\\ \end{bmatrix} \quad$ | W_3  matrix size is m x n, where, m = number of previous layer neurons  and n = number of neurons in the current layer(output layer). |
| $\large b_3$            | $\large\begin{bmatrix} b_{31}  \\ b_{32}  \\ ...  \\ b_{3n} \\ \end{bmatrix} \quad$ | b_2 is a vector with size as n, number of neuron in layer 2. |
| $\large a_{3n}$         | $\large \begin{bmatrix} a_{31}  \\ a_{32}  \\ ...  \\ a_{3n} \\ \end{bmatrix} \quad = W_3 * h_2 + b_3 $ | A pre-activation function vector of size n, number of neurons in output layer. The Matrix multiplication size will be [n * m ] * [m * 1] + [n*1] = [n * 1]. |
| $\large h_{3n}$         | $\large \begin{bmatrix} h_{31}  \\ h_{32}  \\ ...  \\ h_{3n} \\ \end{bmatrix} \quad = \begin{bmatrix} softmax(a_{31})  \\ softmax(a_{32})  \\ ...  \\ softmax(a_{3n}) \\ \end{bmatrix} \quad$ | A activation function vector of size n, number of neurons in output layer. The activation can be any non-linear continous function like Sigmoid, tanh etc. Here we will use softmax function. |

For softmax, 



$ \Large  softmax(a_{31})  = \LARGE \frac{e^{a_{31}}}{\sum e ^ {a_i}}$ where *i*, ranges from 1 to n, n being the number of neurons at the output layer. 



So if you combine the above layer equations it will result into below:



$\large \hat{y} = f(x) = O(W_3g(W_2g(W_1X_1+b_1) + b_2)+ b_3)$, where $O$ can be a softmax or any other non-linear continous function. 



CROSS ENTROPY LOSS FUNCTION  for Multi-Class Classification is given by:



$\large L(\Theta)  = -[(1-y)log(1 - \hat{y}) + ylog\hat{y}] $







