# RAGHU_BATCH_4_ASSIGNMENT2B

## Assignment 2A git hub link:
[RAGHU_BATCH_4_ASSIGNMENT2A.ipynb](https://github.com/rraghu214/MLBLR/blob/master/RAGHU_BATCH_4_ASSIGNMENT2A.ipynb)

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
```

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
 wh=np.random.random((4,3))
np.set_printoptions(precision=4,suppress=True )
print("Weight is \n")
print(wh)

bh=np.random.random((1,3))
print("\n Bias is\n")
print(bh)
 ```
 
 Weight is 

[[0.0886 0.7621 0.5902]
 [0.2421 0.8832 0.9173]
 [0.9116 0.3707 0.3448]
 [0.9823 0.0808 0.563 ]]

 Bias is

[[0.4923 0.4771 0.8014]]

#### Step 2: Calculate hidden layer input:
hidden_layer_input = matrix_dot_product(X,wh) + bh
```python
from numpy.core.multiarray import ndarray

hidden_layer_input: ndarray = X.dot(wh) + bh
print(hidden_layer_input)
```
[[1.4925 1.6098 1.7364]
 [2.4748 1.6906 2.2994]
 [1.7167 1.441  2.2817]]
 
#### Step 3: Perform non-linear transformation on hidden linear input
 hiddenlayer_activations = sigmoid(hidden_layer_input)
 
```python
def sigmoid(h):
   return 1 / (1 + np.exp(-h))

hiddenlayer_activations = sigmoid(hidden_layer_input)
print(hiddenlayer_activations)
```

[[0.8165 0.8334 0.8502]
 [0.9224 0.8443 0.9088]
 [0.8477 0.8086 0.9074]]
 
 #### Step 4: Perform linear and non-linear transformation of hidden layer activation at output layer
 
 output_layer_input = matrix_dot_product (hiddenlayer_activations * wout ) + bout 
 output = sigmoid(output_layer_input)
 
 ```python
 wout=np.random.random((3,1))
np.set_printoptions(precision=4,suppress=True )
print("Weight at hidden layer\n")
print(wout)
print("\n Bias at hidden layer")
bout=np.random.random((1,1))
print(bout)

output_layer_input = hiddenlayer_activations.dot(wout) + bout
print("\n Output_layer_input \n")
print(output_layer_input)

output=sigmoid(output_layer_input)
print("\n Output \n")
print(output)
 ```
 
 Weight at hidden layer

[[0.7673]
 [0.2478]
 [0.2463]]

 Bias at hidden layer
[[0.7906]]

 Output_layer_input 

[[1.8329]
 [1.9313]
 [1.8648]]

 Output 

[[0.8621]
 [0.8734]
 [0.8659]]
 
#### Step 5: Calculate gradient of Error(E) at output layer

E = y-output

```python
E= y - output
print(E)
```

[[ 0.1379]
 [ 0.1266]
 [-0.8659]]
 
 
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

[[0.1189]
 [0.1106]
 [0.1161]]

 Slope_hidden_layer 

[[0.1499 0.1389 0.1273]
 [0.0716 0.1315 0.0829]
 [0.1291 0.1548 0.0841]]
 
#### Step 7: Compute delta at output layer

d_output = E * slope_output_layer*lr

```python
lr = 0.1

d_output = E * Slope_output_layer * lr
print(d_output)
```
[[ 0.0016]
 [ 0.0014]
 [-0.0101]]

#### Step 8: Calculate Error at hidden layer

Error_at_hidden_layer = matrix_dot_product(d_output, wout.Transpose)

```python
Error_at_hidden_layer=d_output * wout.transpose()
print(Error_at_hidden_layer)
```

[[ 0.0013  0.0004  0.0004]
 [ 0.0011  0.0003  0.0003]
 [-0.0077 -0.0025 -0.0025]]
 
#### Step 9: Compute delta at hidden layer

d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

```python
d_hiddenlayer=Error_at_hidden_layer*Slope_hidden_layer
print(d_hiddenlayer)
```

[[ 0.0002  0.0001  0.0001]
 [ 0.0001  0.      0.    ]
 [-0.001  -0.0004 -0.0002]]
 
 
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

[[0.7667]
 [0.2472]
 [0.2456]]

 wh is 

[[0.0887 0.7621 0.5902]
 [0.242  0.8832 0.9173]
 [0.9116 0.3707 0.3448]
 [0.9822 0.0807 0.563 ]]


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

[[0.4922 0.477  0.8014]]

 bout is 

[[0.7899]]
