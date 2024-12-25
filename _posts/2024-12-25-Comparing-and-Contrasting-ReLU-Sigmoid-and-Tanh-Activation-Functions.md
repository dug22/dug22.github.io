---
title:  "Comparing and Contrasting ReLU Sigmoid and Tanh Activation Functions"
categories:
  - machine learning
tags:
  - machine learning
  - data science
  - python
---

# **Introduction**
<p align="justify">
Activation functions play a very important role in deep learning because they determine how neurons in a given neural network process inputs to produce outputs. ReLU, sigmoid, and tahn (hyperbolic tangent) are some of the most commonly used activation functions in deep learning. Each of these activation functions possesses unique characteristics that make them well-suited for specific scenarios. In this article, we are going to explore what these activation functions are and how they are used in deep learning.

</p>

# **What is an Activation Function?**

<p align= "justify">
An activation function is a mathematical equation that determines the output of a neuron in a neural network based on its input. Without action functions, neural networks would be limited to modeling only linear relationships between inputs and outputs. 
Selecting the appropriate activation function is essential for training neural networks, ensuring they generalize well and produce accurate predictions. There are a lot of activation functions to choose from, but its essential for the user to choose the right one to use 
for a given neural network.
</p>

  
# **ReLU**

ReLU or Rectified Linear Unit, operates by outputting the input directly if the input is greater than or equal to
zero, and ouputting 0 if the input is less than zero. The given formula for ReLU is defined as:

* f(x) = max(0,x)
  * x resembles the input value

**Advantages of ReLU**
* ReLU mitigates the vanishing gradient problem. For positive inputs, its gradient is constant.
* ReLU promotes sparsity by setting negative values to zero.

**Disadvantages of ReLU**
* The dying ReLU problem can occur. This problem occurs when a ReLU neuron consistently receives negative inputs during training, and consistently outputting zero.
* Unbounded outputs can arise when ReLU activations produce very large values, potentially leading to exploding gradients. 

Here is a simple ReLU activation function plot using Python:

~~~python
import plotly.graph_objects as go
import numpy as np

def relu(x):
    return np.maximum(0, x)

x_values = np.linspace(-10, 10, 400)
y_values = relu(x_values)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='ReLU Curve'))
fig.update_layout(
    title="ReLU Function",
    xaxis_title="x",
    yaxis_title="ReLU(x)",
    showlegend=True,
    width=1000,
    height=800
)

fig.show()
~~~

![](https://i.imgur.com/PPUOuim.png)

# **Sigmoid**
A sigmoid is a mathematical curve that transforms input values into an output ranging between 0 and 1, 
characterized by its distinct S shaped graph. The given formula for sigmoid is defined as:

* σ ( x ) = 1 1 + e − x 
  * x resembles the input value.
  * e is Euler's number which equals 2.71828.

**Advantages of Sigmoid**
* It gives a smooth gradient that prevents jumps in output values.
* It is one of the best normalized functions.
* Outputs range for sigmoid is (0,1), making it an ideal activation function for binary classification tasks.

**Disadvantages of Sigmoid**
* Vanishing Gradient Problem.
  * The gradient of the sigmoid function can become very small due to large positive or negative input values, which could slow down the learning process for our neural network.
* Non zero-centered ouput.
  * The output of the sigmoid functions ranges from 0 to 1, meaning not zero centered, which can lead to inefficient gradient updates in optimization.

Here is a simple sigmoid activation function plot using Python:

~~~python
import plotly.graph_objects as go
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_values = np.linspace(-10, 10, 400)
y_values = sigmoid(x_values)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Sigmoid Curve'))
fig.update_layout(
    title="Sigmoid Function",
    xaxis_title="x",
    yaxis_title="Sigmoid(x)",
    showlegend=True,
    width=1000,
    height=800
)

fig.show()
~~~

![](https://i.imgur.com/H1gW7aH.png)

# **Tahn (hyperbolic tangent)**
The tahn function outputs values in the range of -1 to +1, making it more effective at handling negative values compared to the sigmoid function, which ranges from 0 to 1. This property helps in centering data around zero, which can accelerate convergence during optimization. The given formula for tahn can be defined as:  

* tahn(x) = (e^x - e^-x) / (e^x + e^-x)
  * x represents the given input
  * The numerator makes tahn(x) positive or negative depending on the input (x).
  * The denominator scales the result, ensuring the output is always within the range -1 to 1.

**Advantages of Tahn**
* The ⁡tanh function is zero-centered, ensuring that the gradients are more symmetrically distributed around 0. Although vanishing or exploding gradients can still occur, this symmetry helps mitigate these issues, resulting in a more stable and efficient training process compared to the sigmoid function.

**Disadvantages of Tahn**
* Similar to sigmoid, tahn can experience the vanishing gradient problem or exploding gradients when the input becomes too large.

Here is a simple tahn activation function plot using Python:

~~~python
import plotly.graph_objects as go
import numpy as np

def tanh(x):
    return np.tanh(x)

x_values = np.linspace(-10, 10, 400)
y_values = tanh(x_values)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Tanh Curve'))
fig.update_layout(
    title="Tanh Function",
    xaxis_title="x",
    yaxis_title="Tanh(x)",
    showlegend=True,
    width=1000,
    height=800
)

fig.show()
~~~

![](https://i.imgur.com/ihkq0s6.png)

# **Conclusion**

<p align="justify">
Selecting the appropriate activation function is essential for optimizing the performance of a neural network. There are various activation functions available; sigmoid and tanh are effective for shallow networks and tasks such as binary classification, while ReLU has become the default choice for deep networks due to its simplicity and efficiency. 
By understanding the strengths and limitations of each activation function, you can better optimize your models for faster convergence and enhanced performance.
</p>

# **References**
* All Moez, Introduction to Activation Functions in Neural Networks. 2024 https://www.datacamp.com/tutorial/introduction-to-activation-functions-in-neural-networks
* Antoniadis Panagiotis, Activation Functions: Sigmoid vs Tahn. 2024 https://www.baeldung.com/cs/sigmoid-vs-tanh-functions
* Neural networks: Activation functions. https://developers.google.com/machine-learning/crash-course/neural-networks/activation-functions

# **GitHub Repository**
You can find the code for this exercise [Here](https://github.com/dug22/datascience-blog-exercises/tree/main/2.Comparing-and-Contrasting-ReLU-Sigmoid-and-Tanh-Activation-Functions.md)
