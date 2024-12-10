## Understading The Perceptron Model

## Introduction 
Rosenblatt's Percepton model is know to be a single unit of logic in artificial intelligence. Rosenblatt's Perceptron model primary use is for binary classification tasks and determine two possible outcomes (1 or 0, yes or no, etcetera).
While the Perceptron has its limitations, its simplicity provides a strong foundation for understanding more complex models. We'll be going over the fundamentals of how Rosenblatt's Perceptron model works in Python using Google Colab.

Here is a given visual of what Rosenblatt's Perceptron model looks like:
![Image](https://private-user-images.githubusercontent.com/190862800/393441348-d5551942-d63f-4f2d-a4a3-46c31e3d8b7c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzM3NDMxMzMsIm5iZiI6MTczMzc0MjgzMywicGF0aCI6Ii8xOTA4NjI4MDAvMzkzNDQxMzQ4LWQ1NTUxOTQyLWQ2M2YtNGYyZC1hNGEzLTQ2YzMxZTNkOGI3Yy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMjA5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTIwOVQxMTEzNTNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mMTg1MzEzYTAwODk1MGY1MWJiOTE4MjFjMmU3MTRlNjFkYWFlZTZlZmJkMzY0MThlZGMxZGZiZWYzZjNjYjVlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.Qs1VnuwK-Gv0_APABtGd4wlzpTZVpq3YLmFCegJVVoM)
## Imports Needed
Before we get started we need to import the dependencies needed for this exercise.
  * Pandas
    
~~~
import pandas
~~~
## Preparing Perceptron Inputs & Weights
<p align="justify">
In order for our Perceptron model to work we need to create a dataset of inputs and weights within our Pandas dataset. We have a set of inputs known as <b>x0</b>,<b>x1</b>, and <b>x2</b> each assigned with a corresponding value. Similarly we have a set
of weights, <b>w0</b>, <b>w1</b>, and <b>w2</b>, each assigned with a corresponding value. These weights dictate the influence each input value has on the overall outcome. Below represents our Pandas dataframe of inputs and weights.
</p>

~~~python
data = {
"Inputs" : [0.6, 0.4, 0.4],
"Weights" : [0.4, 0.3, 0.3
]}
~~~

## Multiplying Corresponding Inputs & Weights & Summing Corresponding Products
<p align="justify">
The input <b>x0</b>​, with a weight of <b>0.4</b>, is the most significant since its weight is greater than those of <b>x1</b>​ and <b>x2</b>​. To reach a logical conclusion, we multiply each input by its corresponding weight and then calculate the sum of these products.
In order to do that in Python, our code uses list comprehension multiply each element in the "<b>Inputs</b>" list by the corresponding elements in the "<b>Weights</b>**" list from the given <b>data</b> dictionary. The given results are then stored in
the variable <b>corresponding_products</b> (we need this variable to find the sum of the given corresponding products). 
</p>

~~~python
corresponding_products = [input_value * weight_value for input_value, weight_value in zip(data["Inputs"], data["Weights"])]
print("Corresponding Products:",corresponding_products)
~~~

```
Output: Corresponding Products: [0.24, 0.12, 0.12]
```
<p align="justify">
Next, take the array of corresponding products and find the sum of it. In this case the sum of the corresponding products is <b>0.48</b>
</p>

~~~python
sum_of_products = sum(corresponding_products)
print("Sum of Corresponding Products:", sum_of_products)
~~~

```
Output: Sum of Corresponding Products: 0.48
```

## Applied Activation Function
<p align="justify">
Next, we check if the given sum exceeds a certain threshold, also known as the bias. To evaluate whether the threshold (also known as the bias) has been met, we use an activation function called the step function. In machine learning, a step function is commonly used where
it outputs a fixed value (0 or 1) based on whether the input crosses a specific threshold. In simpler terms:
</p>

```
f(x) = 1, if x > bias  0, otherwise
```
<p align="justify">
We need to create a step function method in Python, and that is really simple. Our bias/threshold value will be <b>0.5</b>.
</p>

~~~python
def step_function(x, bias):
  return 1 if x > bias else 0
~~~

Now let's pass our **sum_of_products** variable, and a bias value of **0.5** into the **step_function** method to observe the outcome.

~~~python
step_function_result = step_function(sum_of_products, 0.5)
print("Step Function Result:", step_function_result)
~~~

```
Output: Step Function Result: 0
```

Our outcome returned **0** because our sum of **0.48** is not greater than our bias value (**0.5**).

## Conclusions
<p align="justify">
Rosenblatt's Perceptron is widely used for binary classification, but its limitations arise from being a simple unit of logic. Despite this, the Perceptron model serves as an excellent introduction to how neural networks make decisions. By grasping key concepts such as inputs, weights, bias, and the summation and activation steps, you build a solid foundation for exploring more complex models, such as the Multilayer Perceptron.
</p>

## Source To Repository
[Understanding The Perceptron Model GitHub Repository](https://github.com/dug22/Understanding-The-Perceptron-Model)

## References 
[What is a Perceptron? – Basics of Neural Networks](https://towardsdatascience.com/what-is-a-perceptron-basics-of-neural-networks-c4cfea20c590)

[Perceptron Algorithm with Code Example - ML for beginners! ](https://www.youtube.com/watch?v=-KLnurhX-Pg)
