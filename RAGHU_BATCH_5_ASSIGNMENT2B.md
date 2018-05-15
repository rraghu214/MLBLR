# RAGHU_BATCH_5_ASSIGNMENT2B

## Assignment 2A git hub link:
[RAGHU_BATCH_5_ASSIGNMENT2A.ipynb](https://github.com/rraghu214/MLBLR/blob/master/RAGHU_BATCH_5_ASSIGNMENT2A.ipynb)

## Assignment 2B: Neural Network Computation 
#####  -- Forward Propogation and Back Propogation

#### Step 0: Read input and output

``` python
import numpy as np
X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]], dtype=int)
print("X is \n")
print((X))
y=np.array([[1],[1],[0]])
print("\n y is \n")
print(y)
```[RAGHU_BATCH_5_ASSIGNMENT2B.md](\:storage\0.tzbfoj06oel.md)

X is 

[[1 0 1 0]
 [1 0 1 1]
 [0 1 0 1]]


 y is 

[[1]
 [1]
 [0]]
 
 Note: Input -X has 3 samples and 4 channels. Hence 3 x 4
 
 #### Step 1: Initialize weights and biases with random values
 
 ```python
wh=np.random.uniform(-1,1,[4,3])
np.set_printoptions(precision=4,suppress=True )
print("Weight is \n")
print(wh)
bh=np.random.uniform(-1,1,[1,3])
print("\n Bias is\n")
print(bh)
 ```
 
Weight is 

[[ 0.4651 -0.0258  0.4467]
 [ 0.9887  0.4021 -0.3974]
 [-0.4321  0.5223  0.0534]
 [-0.5103 -0.7675 -0.582 ]]

 Bias is

[[ 0.4327 -0.7248  0.0655]]

#### Step 2: Calculate hidden layer input:
hidden_layer_input = matrix_dot_product(X,wh) + bh
```python
from numpy.core.multiarray import ndarray

hidden_layer_input: ndarray = X.dot(wh) + bh
print(hidden_layer_input)
```
[[ 0.4657 -0.2283  0.5657]
 [-0.0446 -0.9959 -0.0163]
 [ 0.9111 -1.0903 -0.9138]]

 
#### Step 3: Perform non-linear transformation on hidden linear input
 hiddenlayer_activations = sigmoid(hidden_layer_input)
 
```python
def reLu(x):
    x[x<=0] = 0
    return x

hiddenlayer_activations = reLu(hidden_layer_input)
print(hiddenlayer_activations)
```

[[0.4657 0.     0.5657]
 [0.     0.     0.    ]
 [0.9111 0.     0.    ]]
 
 #### Step 4: Perform linear and non-linear transformation of hidden layer activation at output layer
 
 output_layer_input = matrix_dot_product (hiddenlayer_activations * wout ) + bout 
 output = reLu(output_layer_input)
 
 ```python
wout=np.random.uniform(-1,1,[3,1])
np.set_printoptions(precision=4,suppress=True )
print("Weight at hidden layer\n")
print(wout)
print("\n Bias at hidden layer")
bout=np.random.uniform(-1,1,[1,1])
print(bout)

output_layer_input = hiddenlayer_activations.dot(wout) + bout
print("\n Output_layer_input \n")
print(output_layer_input)

output=reLu(output_layer_input)
print("\n Output \n")
print(output)
 ```
 
Weight at hidden layer

[[ 0.0041]
 [ 0.806 ]
 [-0.8683]]

 Bias at hidden layer
[[0.6722]]

 Output_layer_input 

[[0.183 ]
 [0.6722]
 [0.6759]]

 Output 

[[0.183 ]
 [0.6722]
 [0.6759]]

 
#### Step 5: Calculate gradient of Error(E) at output layer

E = y-output

```python
E= y - output
print(E)
```

[[ 0.817 ]
 [ 0.3278]
 [-0.6759]]
 
 
 ###### Summarizing Computation so far


|             X                   ||||              wh            |||         bh            ||| hidden_layer_input ||| hidden_layer_activations ||| wout | bout | output | y | E |
| :---: | :---: | :---: | :---: | :---:| :----: | :---:         | :---: | :---: | :---: | --- | --- | ---           | --- | --- |        ---------  | :---: | :---: | :---: | :---: | :--- |
|   1   |   0    | 1               | 0 |0.4651 | -0.0258  |0.4467 | 0.4327 | -0.7248 | 0.0655 |   0.4657 | -0.2283 | 0.5657   |   0.4657 | 0.   | 0.5657    |  0.0041 | 0.6722 | 0.183 | 1 | 0.817 |
| 1  | 0  | 1  | 1                 |  0.9887 | 0.4021 | -0.3974    |                         |||  -0.0446 | -0.9959 | -0.0163 |    0.   |  0.    | 0.     |  0.806 |    | 0.6722 | 1 | 0.3278 |
| 0  | 1  | 0 | 1                 |-0.4321 | 0.5223 | 0.0534   |                         |||    0.9111 | -1.0903 | -0.9138   |   0.9111 | 0.  |   0.    |  -0.8683 |   | 0.6759 | 0 | -0.6759 |
|  |||| -0.5103 | -0.7675 | -0.582   |


 
 #### Step 6: Compute slope at output and hidden layer

Slope_output_layer= derivatives_sigmoid(output)

Slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)

```python
def slope_calc(S):
    return S*(1-S)

Slope_output_layer=slope_calc(output)
print("\n Slope_output_layer \n")
print(Slope_output_layer)

Slope_hidden_layer=slope_calc(hiddenlayer_activations)
print("\n Slope_hidden_layer \n")
print(Slope_hidden_layer)
```

 Slope_output_layer 

[[0.1495]
 [0.2203]
 [0.219 ]]

 Slope_hidden_layer 

[[0.2488 0.     0.2457]
 [0.     0.     0.    ]
 [0.081  0.     0.    ]]

 
 |Slope_hidden_layer |||Slope_output_layer  |
 | :---: | :--: | :---: | :--: |  :--: |
|0.2488 | 0.  |   0.2457 | 0.1495 |
 |0.  |   0.  |   0.    | 0.2203 |
 |0.081 |  0.  |   0.    |  0.219 |
#### Step 7: Compute delta at output layer

d_output = E * slope_output_layer*lr

```python
lr = 0.1

d_output = E * Slope_output_layer * lr
print(d_output)
```
[[ 0.0122]
 [ 0.0072]
 [-0.0148]]



|delta output|
| :--: |
|0.0122|
 | 0.0072|
 |-0.0148|


#### Step 8: Calculate Error at hidden layer

Error_at_hidden_layer = matrix_dot_product(d_output, wout.Transpose)

```python
Error_at_hidden_layer=d_output * wout.transpose()
print(Error_at_hidden_layer)
```

[[ 0.      0.0098 -0.0106]
 [ 0.      0.0058 -0.0063]
 [-0.0001 -0.0119  0.0129]]
 
 |error at hidden layer|||
 | :--: | :--: | :--: |
| 0.     | 0.0098 | -0.0106|
 | 0.    |  0.0058  | -0.0063|
 |-0.0001 | -0.0119 | 0.0129|
 
#### Step 9: Compute delta at hidden layer

d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

```python
d_hiddenlayer=Error_at_hidden_layer*Slope_hidden_layer
print(d_hiddenlayer)
```

[[ 0.      0.     -0.0026]
 [ 0.      0.     -0.    ]
 [-0.     -0.      0.    ]]

 
 |delta at hidden layer |||
 | :--: | :--: | :--: |
| 0.    |  0.  |   -0.0026|
 | 0.   |   0. |    -0.    |
 |-0.   | -0.  |    0.    |
 
#### Step 10: Update weight at both output and hidden layer

wout = wout + matrix_dot_product (hiddenlayer_activations.Transpose, d_output) * learning_rate

wh = wh+ matrix_dot_product (X.Transpose,d_hiddenlayer) * learning_rate

```python
wout=wout+(hiddenlayer_activations.transpose().dot(d_output)) * lr
print("\n wout is \n")
print(wout)

wh=wh+X.transpose().dot(d_hiddenlayer) * lr
print("\n wh is \n")
print(wh)
```

 wout is 

[[ 0.0033]
 [ 0.806 ]
 [-0.8676]]

 wh is 

[[ 0.4651 -0.0258  0.4464]
 [ 0.9887  0.4021 -0.3974]
 [-0.4321  0.5223  0.0532]
 [-0.5103 -0.7675 -0.582 ]]


#### Step 11: Update biases at both output and hidden layer

bh = bh + sum(d_hiddenlayer, axis=0) * learning_rate

bout = bout + sum(d_output, axis=0)*learning_rate


```python
bh = bh + np.sum(d_hiddenlayer, axis=0) * lr
print("\n bh is \n")
print(bh)
print("\n bout is \n")
bout = bout + np.sum(d_output, axis=0) * lr
print(bout)
```

 bh is 

[[ 0.4327 -0.7248  0.0653]]

 bout is 

[[0.6727]]



### Back Prop values with updated weights and biases.

|             X                   ||||              wh            |||         bh            ||| hidden_layer_input ||| hidden_layer_activations ||| wout | bout | output | y | E |
| :---: | :---: | :---: | :---: | :---:| :----: | :---:         | :---: | :---: | :---: | --- | --- | ---           | --- | --- |        ---------  | :---: | :---: | :---: | :---: | :--- |
|   1   |   0    | 1               | 0 |0.4651 | -0.0258  |0.4464 | 0.4327 | -0.7248 | 0.0653 |   0.4657 | -0.2283 | 0.5657   |   0.4657 | 0.   | 0.5657    |  0.0033 | 0.6722 | 0.183 | 1 | 0.817 |
| 1  | 0  | 1  | 1                 |  0.9887 | 0.4021 | -0.3974    |                         |||  -0.0446 | -0.9959 | -0.0163 |    0.   |  0.    | 0.     |  0.806 |    | 0.6727 | 1 | 0.3278 |
| 0  | 1  | 0 | 1                 |-0.4321 | 0.5223 | 0.0532  |                         |||    0.9111 | -1.0903 | -0.9138   |   0.9111 | 0.  |   0.    |  -0.8676 |   | 0.6759 | 0 | -0.6759 |
|  |||| -0.5103 | -0.7675 | -0.582   |