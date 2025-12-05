# Cat Breed Classification
Authors: Sandy Tam, Christina Vo
<br>
### Description of Research Topic
This project aims to explore how can we utilize machine learning models, CNN and logisitic regression, to distinguish between different cat breeds. By collecting, labeling, and training on a collected dataset, we will investigate what balance of dataset size, complexity, and generalization leads to the best performing model. Furthermore, we will analyze which visual features and data characteristics most influence model decisions. 
<br>

### Project Outline
For our project outline, we will first search for our datasets, specifically looking for cat breed datasets containing images and characteristics. After having our datasets, we planned on testing the datasets with two models, CNN and Logistic Regression for two datasets to see which model would perform the best.
<br>

### Data Collection Plan
Sandy: I will gather high quality, diverse cat breed data from various sources such as Kaggle. Then, I will process the data to ensure consistency and split it into training and testing sets. I plan to use data augmentation techniques to increase the diversity of the dataset. This step is to prevent overfitting and allow the CNN to distinguish subtle differences between the different cat breeds. I will also manually procure and process some images for the validation set.

<br>
Christina: When looking for cat breed datasets, there are two different data I want to look for. For one dataset, it would need to contain images in order to compare different cat breeds based on visuals. As for the other dataset, it would need to contain the different characteristics of cat breeds to find the relationships in order to identify the cat breed.

### Model Plans
Sandy: I plan to design and implement a simple CNN using PyTorch. The initial model will consist of two layers, followed by pooling layers, fully connected layers, and a softmax output layer. I will experiment with various kernel sizes, activation functions, and optimizers to find the best performing model. To address training efficiency and concerns of overfitting, I will incorporate batch normalization and dropout layers. The model's performance will be evaluated using metrics such as accuracy and F1-score.

<br>
Christina: I plan on using a Logistic Regression model for the characteristics dataset as it can help in finding the relationships between features such as fur color, fur length, body weight, and other traits to identify the breed. I would test the data with train-test splitting and normalization to ensure that the model performs well. 

### CNN Model Dependencies
To run the CNN model, you'll need the following Python packages:

#### Core Dependencies
- **Python 3.x** (recommended 3.8 or higher)
- **PyTorch** - Deep learning framework
  ```bash
  pip install torch torchvision
  ```
- **NumPy** - Numerical computing
  ```bash
  pip install numpy
  ```
- **Matplotlib** - Plotting and visualization
  ```bash
  pip install matplotlib
  ```

#### Optional Dependencies (for GPU acceleration)
If you have a compatible NVIDIA GPU and want to use CUDA acceleration:
- CUDA Toolkit (compatible with your PyTorch version)
- cuDNN

To install PyTorch with CUDA support, visit [PyTorch's official website](https://pytorch.org/get-started/locally/) and select your configuration.

### How to Run the CNN Model

#### 1. Run Preprocessing.ipynb to Prepare Your Data
Run the preprocessing notebook and ensure your data is organized in the following structure:
```
CNN/
├── data/
│   ├── train/
│   │   ├── breed1/
│   │   ├── breed2/
│   │   └── ...
│   └── test/
│       ├── breed1/
│       ├── breed2/
│       └── ...
```

#### 2. Configure Training Parameters
Open `cnn.ipynb` and adjust the hyperparameters as needed:
- `BATCH_SIZE`: Number of samples per batch (default: 32)
- `LEARNING_RATE`: Learning rate for optimizer (default: 0.0001)
- `NUM_EPOCHS`: Number of training epochs (default: 50)
- `KERNEL_SIZE`: Convolutional kernel size (default: 5)
- `TARGET_SIZE`: Image resize dimensions (default: 224x224)

#### 3. Run the Model
You can run the CNN model in two ways:

**Option A: Using Jupyter Notebook** (Recommended)
1. Navigate to the CNN directory
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `cnn.ipynb`
4. Run all cells sequentially

**Option B: Converting to Python Script**
If you prefer to run it as a script:
1. Convert the notebook to a Python script:
   ```bash
   jupyter nbconvert --to script cnn.ipynb
   ```
2. Run the script:
   ```bash
   python cnn.py
   ```

#### 4. Resume Training from Checkpoint
The model automatically saves checkpoints during training. To resume from the last checkpoint:
- Set `START_FRESH = False` in the configuration cell
- The model will automatically load the most recent checkpoint from the `checkpoints/` directory

#### 5. View Training Results
After training completes:
- The trained model is saved as `cat_breed_cnn.pth`
- Training history is saved as `training_history.json`
- Checkpoints are saved in the `checkpoints/` directory
- Training/validation loss plots are displayed in the notebook

#### 6. GPU vs CPU Training
The model automatically detects and uses available hardware:
- **GPU (CUDA/MPS)**: If available, training will use GPU acceleration
- **CPU**: Falls back to CPU if no GPU is detected

To force CPU usage or check device:
```python
device = torch.device("cpu")  # Force CPU
# or
print("Using device:", device)  # Check current device
```

#### Troubleshooting
- **Out of Memory Error**: Reduce `BATCH_SIZE`
- **Slow Training**: Check if GPU is being utilized, or reduce image resolution
- **Poor Performance**: Try adjusting learning rate, increasing epochs, or experimenting with different kernel sizes
- **Checkpoint Issues**: Delete the `checkpoints/` folder to start completely fresh


### Logistic Regression Dependencies
To run the Logistic Regression model, you will need the following packages:

### Core Dependencies
- __Python 3.x__ (recommended 3.8 or higher)
- __NumPy__
  ```bash
  pip install numpy
  ```
- __Matplotlib__
  ```bash
  pip install matplotlib
  ```
- __Pandas__
  ```bash
  pip install pandas
  ```
- __Scikit-learn__
- ```bash
  pip install sklearn
  ```

### 1. How to Run the Logistic Regression
Run the preprocessing.ipynb and ensure that the cleaned dataset is in the following structure:
```
LogisticRegression/
├── data/
│   ├── cat_breeds_dirty
│   ├── cat_breeds_clean
│       
│       
│   
```

### 2. Configure the Training Parameters
Open Cat Breed Classifier (LR).ipynb and changed the parameters to your liking to test the model
- `learning_rate`: learning rate of optimizer (default: 0.1)
- `n_iters`: numbers of iterations (deafult: 1000)
- `random_seed`: random number generator (default: 1)

### 3. Run the Model
Run the model using Jupyter Notebook
1. Navigate to the LogisticRegression directory
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `Cat Breed Classifier (LR).ipynb`
4. Run all cells sequentially

### 5. Review Results
Results will be under the last call of Cat Breed Classifier (LR).ipynb

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

If we were to continue working on this project, we would pursue the following enhancements:

#### CNN Improvements
- **Transfer Learning**: Implement pre-trained models (ResNet, VGG16, EfficientNet) and fine-tune them on our cat breed dataset to leverage learned features from larger datasets
- **Attention Mechanisms**: Add attention layers to help the model focus on distinguishing features like ear shape, face structure, and fur patterns
- **Better Dataset**: Expand the dataset with more images per breed, use higher quality images, and include rarer breeds
- **Web Interface**: Create a user-friendly web application for breed prediction with confidence scores
- **Age and Gender Prediction**: Extend model to predict additional cat characteristics

#### Logisitic Regression Improvements
