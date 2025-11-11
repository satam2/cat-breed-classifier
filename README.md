# Cat Breed Classification
Authors: Sandy Tam, Christina Vo
<br>
### Description of Research Topic
This project aims to explore how can we utilize machine learning models, CNN and logisitic regression, to distinguish between different cat breeds using only image data. By collecting, labeling, and training on a collected dataset, we will investigate what balance of dataset size, complexity, and generalization leads to the best performing model. Furthermore, we will analyze which visual features and data characteristics most influence model decisions. 
<br>

### Project Outline
For our project outline, we will first search for our datasets, specifically looking for cat breed datasets containing images and characteristics. After having our datasets, we planned on testing the datasets with two models, CNN and Logistic Regression for two datasets to see which model would perform the best.
<br>

### Data Collection Plan
Sandy: I will gather high quality, diverse cat breed data from various sources such as Kaggle. Then, I will process the data to ensure consistency and split it into training and testing sets. I plan to use data augmentation techniques to increase the diversity of the dataset. This step is to prevent overfitting and allow the CNN to distinguish subtle differences between the different cat breeds. I will also manually procure and process some images for the validation set.

<br>
Christina: When looking for cat breed datasets, there are two different data I want to look for. For one dataset, it would need to contain images in order to compare different cat breeds based on visuals. As for the other dataset, it would need to contain the different characteristics of cat breeds to find the relationships in order to identify the cat breed.

### Model Plans
Sandy: I plan to design and implement a simple CNN using PyTorch. The initial model will consist of two layers, followed by pooling layers, fully connected layers, and a softmax output layer. I will experiment with various kernel sizes, activatoin functions, and optimizers to find the best performing model. To address training efficiency and concerns of overfitting, I will incorporate batch normalization and dropout layers. The model's performance will be evaluated using metrics such as accuracy and F1-score.

<br>
Christina: I plan on using a Logistic Regression model for the characteristics dataset as it can help in finding the relationships between features such as fur color, fur length, body weight, and other traits to identify the breed. I would test the data with train-test splitting and normalization to ensure that the model performs well. 

### Project Timeline
**Week of 10/21 and 10/23** - Search for Datasets
<br>
**Week of 10/28 and 10/30** - Clean and Preprocess Datasets
<br>
**Week of 11/4 and 11/6** - Code and Train Models
<br>
**Week of 11/11 and 11/13** - Analyze Results
<br>
**Week of 11/18 and 11/20** - Create Presentation and Final Project Meetings
<br>
**Week of 11/25 and 11/27** - Prepare for Project Presentation
<br>
**Week of 12/2 and 12/4** - Final Project Presentations
