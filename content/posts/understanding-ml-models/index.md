---
title: "Understanding Supervised Machine Learning Models"
description: "Understanding how machine learning models work by creating a simple regression model"
date: 2022-07-07T21:34:19+05:30
draft: false
math: true
cover: "/posts/understanding-ml-models/MLModel.svg"
---

A machine learning model functions like any other algorithm in that it takes some input and produces some output base on the input.

<image src="MLModel.svg" class="u-full-width">

The main difference between a machine learning model and a traditional algorithm lies in the way they are created. A supervised machine learning model is created using what is known as training data. The training data is a small collection of inputs and their corresponding outputs that we know to be accurate. The model utilizes the training data to find a common pattern between the inputs and the outputs.

To further understand how a machine learning model is created let us create a very simple model that finds the correlation between a person's height and their shoe size.

> ***Note**: For this example, I am using python and numpy. But you should be able use the same principle in any other language.*

## Predicting shoe size based on height

|
| - | - | - | - | - | - | - | - |
|Input (Height in inches)|56|60|63|64|67|68|70|
|Output (Shoe Size)|7.31|8.25|9.15|9.18|10|10.02|10.81|

If we visualize this data we can see that the relationship between the height and the shoe size is linear.

<image src="height-shoe_size.jpg" class="u-full-width">

So, we can use the equation of a line to create the model:

$$ y = mx + c $$

Here,\
x is the input,\
y is the output and\
m and c are the parameters of the model which we will try to find.

The training process will allow the model to find the parameters ( m and c) of the line.

### Training the model

<image src="TrainingMLModel.svg" class="u-full-width">

The model is trained using a process called **Gradient Descent**. Gradient Descent is a method to find the parameters which minimize the error of our model. It works by progressively updating the parameters of the model based on the error of the model.

The **Cost Function** tells us how far off the predicts of our model are from the training data.

We can break down the training process into the following steps:

1. Make predictions for the training data.
2. Calculate the cost using the cost function.
3. Update the parameters of the model using the cost.

To train our model we repeat the above steps until we are satisfied with the model. So lets get started!

#### Initializing the parameters

We start by initializing the parameters ( m and c ) of the model.
We can either initialize them to random values or we can initialize them to zero.

For this example, we will just initialize them to zero.
```python
m = 0
c = 0
```

#### Defining the model

To make the predicts the use the line equation as discussed above:
$$ y = mx + c $$

```python
def predict(x):
    return m * x + c
```

This process is also called **forward propagation**.

#### Calculating the cost

To calculate the cost for our linear regression model, we can use one of the following cost functions:

1. Mean Absolute Error (MAE)
$$ \frac 1 n \sum_{i=1}^n |y - \hat y|$$
2. Mean Squared Error (MSE)
$$ \frac 1 n \sum_{i=1}^n (y - \hat y)^2 $$

> ***Note**: y refers to the actual value and y hat refers to the predicted value*

For this example we are going to use the Mean Squared Error (MSE) cost function.

```python
def cost(y, y_hat):
    return np.sum((y - y_hat) ** 2) / len(y)
```

#### Updating the parameters


