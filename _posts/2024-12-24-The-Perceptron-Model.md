---
title:  "The Perceptron Model"
categories:
  - machine learning
tags:
  - machine learning
  - data science
  - python
---

In this article, we’ll explore the fundamentals of the perceptron model and walk through the process of creating and training our own perceptron model from scratch using Python.

# **Introduction**

<p align="justify">
The perceptron was created by American psychologist Franklin Rosenblatt in 1957. Rosenblatt set out to study the human brain using brain models,
that were used to explain the brain's capabilities in math and physics. Rosenblatt's first demonstration of the perceptron was done when he used an 
IBM 704, which is a 5-ton computer, the size of a small room. The demonstration involved feeding the perceptron a series of punch cards to differentiate 
between cards marked on the left versus the right. After 50 trials, the computer successfully taught itself how to make out the cards marked on the left 
from the cards marked on the right. 
</p>
<p align="justify">
The perceptron consists of one layer of neurons, which takes in inputs with assigned values, and each input has an assigned weight. Each assigned weight, with an assigned value, can always be readjusted. Both the assigned input, and weight are then multiplied, and once
the corresponding products are found, we need to find the initial sum of the products. The sum is then passed into a given activation function,
to get two possible outcomes. 
</p>

# **Graphing Rosenblatt's Perceptron**
A single perceptron is known to follow a linear pattern. 

Please refere to the two scatter plots below:


![Linear vs Non-Linear](https://i.imgur.com/j47Lgw9.png)

<p align="justify">
In the example above, we are presented with two scatter plots. The first scatter plot follows a linear pattern, while the second scatter plot follows a non-linear pattern. Applying a single perceptron to the first scatter plot works because both outcomes (the green and blue dots) can be separated by a best-fitting line. However, applying a single perceptron to the second scatter plot does not work, as the outcomes are clustered in a way that no line can separate the green and blue dots.
</p>

# **What is the Step Function?**
<p align="justify">
In machine learning, a "step function" is a simple activation function used in neural networks. It outputs either 0 or 1 based on whether the input value is below or above a specified threshold or bias. After calculating the sum of the corresponding products, 
this sum is passed through the step function to determine the outcome. If the sum exceeds the chosen threshold or bias, the outcome is classified as 1; otherwise, it is classified as 0. Let's assume our bias is 0.5 for the example below.
</p>

![](https://i.imgur.com/FXBan6z.png)

# **Single Perceptron in Action**
The table below contains three input values, three corresponding weights, and their respective products.

<center><img src="https://i.imgur.com/RrHiz70.png"></center>


<p align="justify">
The inputs, assigned weights, and the products of x×wx×w are used in the model illustrated below. The calculated sum is 0.31, and with a bias of 0.5, the outcome will be 0. The diagram below demonstrates how the entire model would appear when everything is combined.
</p>

![](https://i.imgur.com/XSxc4Fc.png)

<p align="justify">
By the way on how the model was drawn out, we took our input values and their assigned weights, and multiplied them together to find their products. The sum of the corresponding products came out to be 0.31. Our sum was then passed into the step function. We know 0.31 is not greater than our assigned bias (0.5), which means the output for this model is 0. The weights can always be readjusted to get a different output. 
</p>

# **Moving Forward**
<p align="justify">
Now that you have an idea on how the single-layer perceptron model works, we're going to create our own perceptron model using Python and train it to make predictions. We are going to train it on 100 randomly generated numbers. If the given number is even (excluding 0) it'll be labeled as a 1, otherwise labeled 0 if the given number is odd. 
</p>

# **Imports**
The following libraries will be using for this exercise are:
* Pandas
* Numpy
* Matplotlib

~~~python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
~~~
# **Setting a Random Seed**
To ensure the same sequence of random numbers are generated we are going to set our seed to 1.

~~~python
np.random.seed(1)
~~~

# **The Perceptron Model Class**

~~~python
class PerceptronModel():

  # Initializes the Perceptron Model
  def __init__(self, num_features):
    self.weights = [0.0] * num_features
    self.bias = 0.0
    # num_features = The number of features (input_dimensions) for the dataset.
    # weights = a list of zeros with a size equal to the number of features. During training these weights will be updated.
    # bias = Our threshold is = 0.


  # Trains the given perceptron model using the given dataset.
  def train(self, X, y, learning_rate, num_iterations):
    for iteration in range(num_iterations):
      for i in range(len(X)):
        x = X[i]
        prediction = self.predict(x)
        error = y[i] - prediction
        for j in range(len(self.weights)):
          self.weights[j] += learning_rate * error * x[j]
        self.bias += learning_rate * error
    # x = a list of input data, where each element is a feature vector.
    # y = a list of target labels.
    # learning_rate = a small positive value used to control the step size of weight updates.
    # num_iterations = the number of times the model will iterate over the dataset. Think of it as how many epochs we will train our model.

  # Predicts the output for a given input.
  def predict(self, x):
    sum = self.bias
    for i in range(len(self.weights)):
      sum+= self.weights[i] * x[i]
    return self.step_function(sum)
  # x = a single input.

  # Acts as the activation function for the perceptron
  def step_function(self, sum):
    return 1 if sum > self.bias else 0
  # If our sum is > than our bias our output will be 1, otherwise 0
  
  # Evaluates the accuracy of our perceptron model.
  def accuracy(self, X, y):
    correct = 0
    size_of_X = len(X)
    for i in range(size_of_X):
      x = X[i]
      prediction = self.predict(x)
      if prediction == y[i]:
        correct+=1
    return correct / size_of_X
  # x = a list of input data.
  # y = a list of true labels corresponding to the input data.
~~~

# **Generating Our Dataset**
<p align="justify">
We are going to create a method that will generate 100 random numbers for us. If a number is even (excluding 0) it's feature label will be assigned a value of 1. If a number is odd its feature label will be assigned a value of 0.
</p>

~~~python
def generate_dummy_data(num_samples=100):
  X = [] # Input values
  y = [] # Feature labels

  for i in range(num_samples):
    number = np.random.randint(1, 100)
    x = [number] # x is assigned a number
    y_values = 1 if number % 2 != 0 else 0 # if even assign that number with a feature value of 1, otherwise 0 if its oddd.
    X.append(x)
    y.append(y_values)

  return np.array(X), np.array(y)
~~~
# **Training The Model**

<p align="justify">
This method brings all the components together. It begins by assigning the x and y variables from the dataset in use. Next, it compares the model's predictions with the actual results, and finally, calculates the model's accuracy.
</p>

~~~python
# Method to train the full model
def train_perceptron():
    X, y = generate_dummy_data(num_samples=100)
    model = PerceptronModel(num_features=X.shape[1])
    model.train(X, y, learning_rate=0.1, num_iterations=100)

    predictions = []
    actual_results = []

    # Printing given predictions and compare them with the actual results
    print("Predictions")
    print("-----------------")
    for i in range(len(X)):
        prediction = model.predict(X[i])
        print(f"Sample # {i} Prediction: {prediction}, Actual: {y[i]}")
        predictions.append(prediction)
        actual_results.append(y[i])

    accuracy = model.accuracy(X, y)
    print("-----------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    df = pd.DataFrame({
        'Prediction': predictions,
        'Actual': actual_results
    })

    return df, accuracy
~~~

# **Visualizing Our Accuracy**
Next, we visualize the accuracy:

~~~python
matching_count = sum(df['Prediction'] == df['Actual'])
mismatching_count = sum(df['Prediction'] != df['Actual'])
labels = ['Correct Predictions', 'Incorrect Predictions']
sizes = [matching_count, mismatching_count]
colors = ['lime', 'red'] 
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title(f'Perceptron Prediction Accuracy (Accuracy: {accuracy * 100:.2f}%)')
plt.axis('equal')
plt.show()
~~~

<p align="justify">
The following code creates a simple pie chart that illustrates the number of correct predictions versus incorrect predictions, providing a clear representation of the model's overall accuracy.
</p>


![](https://i.imgur.com/d70zmX0.png)

# **Conclusion**
<p align="justify">
The perceptron is a fundamental block in machine learning, but it has its limitations. The perceptron is mainly used for classification and linearly separable tasks only. Despite the perceptron's limitations, the perceptron has played an important role in the evolution of neural networks, paving the way for models like multi-layer perceptrons 
and deep learning architectures. The perceptron remains as a valuable tool for understanding the basic of supervised learning and binary classification. 
</p>

# **References**
* Sheldon Robert. What is a perceptron? https://www.techtarget.com/whatis/definition/perceptron

* Melanie Lofkowitz. Professor's 'perceptron' paved the way for ai - 60 years too soon. 2019. https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon

* Jean-Christophe B. Loiseau. Rosenblatt’s perceptron, the first modern neural network. 2019 https://towardsdatascience.com/rosenblatts-perceptron-the-very-first-neural-network-37a3ec09038a

# **GitHub Repository**
You can find the code for this exercise [Here](https://github.com/dug22/datascience-blog-exercises/tree/main/1.The%20Perceptron%20Model) 
