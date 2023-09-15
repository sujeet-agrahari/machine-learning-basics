# Machine Learning

- Today, when refer to AI, it is most likely ML only
- AI is a huge set of tools to make computers behave intelligently
- ML is most important subset of AI

## What is ML?

A set of tools to making inferences and predictions from provided data

- Predict future events i.g.
  - Will it rain tomorrow? Yes (75% probability)
- Infer the causes of events and behaviors, it's more about drawing insights e.g.
  - Why does it rain?
- Infer patterns
  - Different types of weather conditions
    - Rainy, sunny etc.
- Inference helps to make predictions

## How does it work?

- It works with the help of statistics and computer science.
- It can learn without step by step instructions.
- It learns patterns from existing data and applies it to new data. i.g. analyzing the spammed emails, it will be able to detect whether an email is spam or not.
- It relies on high quality data

### What is use of Data Science?

Data science is about making discoveries and creating insights from data.

## ML Models

It's a statistical representation of a real-world process on data.

```
 [Cat Image] --> [Cat Detection Model] --> [Yes/No]

```

If we created a model based on traffic details with date wise,

```
[Future Date] --> [Traffic Prediction Model] --> [Traffic Conditions]
```

## Types of ML

1. Reinforcement Learning
2. Supervised Learning
3. Unsupervised Learning

- Training Data
  - existing data to learn from
- Training a Model
  - when a model is being built from training data

### Supervised Learning

![Alt text](/assets/supervised-learning.png)
After the model is trained on above data, we can make predicitons.
![Alt text](/assets/supervised-learning-2.png)

In supervised learning outcomes are known set of values i.e. for above it can either be true or false

### Unsupervised Learning

It doesn't have labels. It is only trained on features.
Usage:

- Anomaly detection
- Clustering or Grouping

![Alt text](/assets/unsupervised-learning.png)
In above, it categorized the data. One category could be high cholesterol and blood sugar in a certain age range.

![Alt text](/assets/unsupervised-learning-2.png)

In reality, data doesn't always come with labels.
It requires manual labour to label.
Label are unknown
No label means, model is unsupervised and finds its own patterns i.e. self driving cars.

## Machine Learning Workflow

![Alt text](/assets/workflow.png)

The above scenario is a example of supervised learning because we already have target label i.e. sale price

### Step 1

![Alt text](</assets/Screenshot 2023-09-13 at 10.38.23 AM.png>)

### Step 2

![Alt text](</assets/Screenshot 2023-09-13 at 10.38.56 AM.png>)

Dataset is split into train and test dataset.

### Step 3

![Alt text](</assets/Screenshot 2023-09-13 at 10.40.09 AM.png>)
![Alt text](</assets/Screenshot 2023-09-13 at 10.40.49 AM.png>)

### Step 4

![Alt text](</assets/Screenshot 2023-09-13 at 10.41.29 AM.png>)
![Alt text](</assets/Screenshot 2023-09-13 at 10.43.26 AM.png>)

If model doesn't perform better i.e. the predictions are not accurate. We need to tune it.
![Alt text](</assets/Screenshot 2023-09-13 at 10.44.29 AM.png>)

### Overall Workflow

![Alt text](</assets/Screenshot 2023-09-13 at 10.45.42 AM.png>)

### Summary of steps

1. Extract features
   - Choosing features and manipulating the dataset
2. Split dataset
   - Train and test dataset
3. Train model
   - Input train dataset into a machine learning model
4. Evaluate
   - If desired performance isn’t reached: tune the model and repeat Step 3

## Supervised Learning

1. Classification
2. Regression

### Classification:

Assigning a category to an observation.

Examples:

- Will this custome stop the subscription? Yes/No
- Is this mole cancerous? Yes/No
- What kind of wine is this? Red/White/Rose
- What is this flower? Rose/Tulip/Lily

### Regression

Assigning a continuous variable

Examples:

- How much this stock be worth?
- What is this exoplanet's mass?
- How tall this child be as an adult?

### Classification Vs Regression

- Regression = continuous
  - Any value within a finite or infinite interval e.g 20F, 20.F
- Classification = category
  - One of few predefined values
    - Cold, Mild, Hot

## Unsupervised Learning

- Unsupervised learning = no target column
  - no guidance
- Looks at the whole dataset
- Tries to detect a pattern

1. Clustering
2. Anomaly Detection
3. Association

### Clustering

Groups the dataset based on their features

Clustering Models:

- K Means: needs to specify the number of clusters
- DBSCAN(density based spatial clustering of applications with noise): don't the number of clusters in advance rather it requires to specify what constitutes the cluster

### Anomaly Detection

Anomaly detection = detecting outliers

outliers = observations that defer from the rest

Usecase:

- Discover devices that fail faster or last longer
- Discover fraudster that manage trick the system
- Discover patients that resist a fatal disease

### Association

Finding relationship between observations.

Events happen together

Usecase:

- Market, Basket analysis - which objects bought together
- People buying beer likely to buy peanuts

## Evaluation

Overfitting:

- performs great on training data
- performs poorly on testing data
- model memorized training data and can't generalize learnings to new data
- using testing data set to check performance

Accuracy = correctly classified observation / all observations

e.g 48 \* 100 / 50 = 96%

Limits of Accuracy: fraud example

![Alt text](</assets/Screenshot 2023-09-13 at 12.03.50 PM.png>)

- False Negative : Telling a pregnant women that you are not pregnant
- False Positive: Telling a man he is pregnant

### Sensitivity

values accurate prediction of fraudulent transactions specifically by valuing true positives more.

![Alt text](</assets/Screenshot 2023-09-13 at 12.07.55 PM.png>)

Above is a bad score, we must consider optimizing it.

### Specificity

which values true negatives. This is useful metric for spam filters. For an email user, it's better to send spam to the inbox rather than send real emails to the spam folder for deletion.

![Alt text](</assets/Screenshot 2023-09-13 at 12.12.55 PM.png>)

## Unsupervised Learning

Choose your own Adventure

## Improving Performance

1. Dimension Reduction
2. Hyperparameter Tuning
3. Ensemble Methods

### Dimension Reduction

Dimension refers to number of features in an observation so dimension reduction means to reducing the number of features.

- Irrelevance: some features don't carry useful information
  - How long it will take us to go to the office
    - Weather, Traffic, but not how many glass of water we drank
- Correlation: some features may be highly related and we can keep and get rid of others
  - hight and shoe size
    - take the hight only
- Collapse: multiple feature in an underlying feature
  - hight and weight
    - body mass index

### Hyperparameter Tuning

![Alt text](</assets/Screenshot 2023-09-13 at 12.24.29 PM.png>)

### Ensemble Methods

Classification Setting:
![Alt text](</assets/Screenshot 2023-09-13 at 12.24.53 PM.png>)
Regression Setting:
![Alt text](</assets/Screenshot 2023-09-13 at 12.25.58 PM.png>)

# Deep Learning

DL uses an algorithm called neural network.

- AKA: Neural Networks, which is loosely inspired by biological neural networks
- Basic Unit: Neurons(Nodes)
- Special area of ML
- But, requires more data than traditional ML
- Best to use when inputs are less structured e.g lar amount of images or texts

**_So, it's like cluster of inter connected nodes(neurons), the nodes can be trained using supervised, unsupervised or reinforcement technique and together they form a Deep Learning model._**

## How does it work?

Let's say you want to predict Box Office revenue for an upcoming movie. You have access to a dataset that maps past movies' box office revenue to their production budget.

![Alt text](</assets/Screenshot 2023-09-15 at 8.22.25 AM.png>)

As you can see, a straight line can be drawn through the data points, showing that as budget goes up, box office revenue increases. This red line is an example of a prediction from a simple model.

The neural network that would accomplish this can be drawn like below. Where budget is passed as input to a neuron that calculates the red curve, and outputs box office revenue.

![Alt text](</assets/Screenshot 2023-09-15 at 8.24.25 AM.png>)

### How does it achieve above?

It forms multiple neurons which has a specific task or use case.

Consider a neuron whose job it is to estimate spend as a function of the budget and the advertising costs.

A second neuron can track how aware people are that the movie has been released. The two things that feed into that are advertising and star power. The more famous your actors, the more aware people are of the movie.
So the second neuron is responsible for awareness.

Lastly, the distribution decisions made by the studio will come into play. Budget, advertising, and timing of the release all feed into another that represents the movie's distribution.

Finally, now that the earlier neurons have figured out the importance of these higher-level concepts, we need to add one more neuron that takes these three factors as an input and outputs the estimated box office revenue. And that's the end of our neural network. Its job is to map relationships between different combinations of variables to the desired output.

![Alt text](</assets/Screenshot 2023-09-15 at 8.31.08 AM.png>)

We just saw an example of a rather small neural network. In reality neural networks are much larger with thousands of neurons. This is the point we start using the term deep learning. By stacking a large number of neurons they can compute incredibly complicated functions that give you very accurate mappings from the input to the output.

So when should we choose Deep Learning as a solution? Deep Learning can outperform other techniques if the data size is large. But with a smaller dataset, traditional machine learning algorithms are preferable. Because of the complexity, deep learning techniques require powerful computers to train in reasonable time. When there is lack of domain knowledge for understanding the features, deep learning outshines traditional machine learning since the neural network figures them out for you. Deep Learning really shines when it comes to complex problems such as computer vision, and natural language processing.

### Major use cases of Deep Learning

- Computer Vision - Image or Video processing
- NLP - (Natural Language Processing) text processing
  e.g language translation, chat bots, personal assistant like alexa or Siri

**Sentiment analysis**

Sentiment Analysis is a Natural Language Processing methodology for quantifying how positive or negative the emotion expressed by a segment of text is. It is often used for automatically categorizing customer feedback messages or product reviews.

# Limits of ML

1. Data Quality
2. Explainability

## Data Quality

![Alt text](</assets/Screenshot 2023-09-15 at 9.58.21 AM.png>)
![Alt text](</assets/Screenshot 2023-09-15 at 9.59.21 AM.png>)
![Alt text](</assets/Screenshot 2023-09-15 at 10.00.33 AM.png>)

## Explainability

![Alt text](<assets/Screenshot 2023-09-15 at 10.08.47 AM.png>)
