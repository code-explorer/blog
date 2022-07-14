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

| Input (Height in inches) | Output (Shoe Size) |
|-|-|
|56|7.31|
|60|8.25|
|63|9.15|
|64|9.18|
|67|10|
|68|10.02|
|70|10.81|

If we visualize this data we can see that the relationship between the height and the shoe size is linear.

<iframe height="700" class="u-full-width" style="border:none;" src="Height_vs_Shoe_Size.html"></iframe>

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

#### Linear Regression

The model we are training is a **Linear Regression** model. A linear regression model tries to find a straight line that best fits the data.

> The same principles used to create a linear regression model also carry over to other supervised machine learning models.

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

During each iteration of the training process, we will update each parameter like so:

$$p = p - LearningRate * \frac{\partial (cost(y, \hat y))}{\partial p} $$

> where p is the parameter and the learning rate controls the rate at which the parameters are updated.

```python
def gradient_descent(y, y_hat, m, c, learning_rate):
    m = m - learning_rate * np.sum((y - y_hat) * x) / len(y)
    c = c - learning_rate * np.sum((y - y_hat)) / len(y)
    return m, c
```

#### Bringing it all together

Now we can bring it all together into a single class like so:

```python
class Regression:
    def __init__(self, x, y):
        self.m = 0
        self.c = 0
        self.x = x
        self.y = y

    def predict(self, x):
        return self.m * x + self.c

    def cost(self):
        y_hat = self.predict(self.x)
        return np.sum((y_hat - self.y) ** 2) / len(self.y)

    def gradient_descent(self, learning_rate):
        y_hat = self.predict(self.x)
        dm = (-2 / len(self.y)) * np.sum((self.y - y_hat) * self.x)
        dc = (-2 / len(self.y)) * np.sum((self.y - y_hat))
        self.m = self.m - learning_rate * dm
        self.c = self.c - learning_rate * dc
        return self.cost()

    def train(self, learning_rate, iterations):
        for i in range(iterations):
            self.gradient_descent(learning_rate)
        return self.m, self.c
```

And finally we can train it by specifying the learning rate and the number of iterations.

```python
model = Regression(x, y)
model.train(1e-5, 1000)
```

<iframe height="700" class="u-full-width" style="border:none;" src="Height_vs_Shoe_Size_No_Normalization.html"></iframe>

There you have it! We have created a regression model that can predict the shoe size based on the height and vice versa.

Now we can take this one step further by normalizing the data so that the input lies between 0 and 1.

We can perform normalization and retrain our model like this:

```python
normalized_x = (x - x.min()) / (x.max() - x.min())

model = Regression(normalized_x, y)
model.train(1e-2, 5000)
```

This gives us a much better model.

<iframe height="700" class="u-full-width" style="border:none;" src="Height_vs_Shoe_Size_After_Regression.html"></iframe>
