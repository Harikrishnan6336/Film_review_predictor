# 🎥 Film_review_predictor

## 📘 Description

A film review predictor that was build based on my study on Deep Convolutional Neural Network for Sentiment Analysis (Text Classification) from [here](https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/).

The dataset used is the Movie review Data which is a collection of movie reviews retrieved from the imdb.com website in the early 2000s by Bo Pang and Lillian Lee. The reviews were collected and made available as part of their research on natural language processing. The reviews were originally released in 2002, but an updated and cleaned up version were released in 2004, referred to as “v2.0”.

You can download the dataset used from [here](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz)

## 🏃‍♂️ Getting Started

#### The 'src' directory contains the following files

* datapreprocess.py - contains all the functions for the processing of the dataset.
* models.py - contains the deep CNN model.
* train.py - contains the code for training the dataset on a deep CNN model.
* predict.py - contains the code for the prediction of the movie review.

### 👨🏻‍🏫  Prerequisites

To install all the dependencies, run:

``` pip install --user -r requirements.txt ```

## 🔧 How to Install

👯 Clone the Repository:
```https://github.com/Harikrishnan6336/movie_review_predictor.git```

Then move to the working directory.

```cd Film_review_predictor```

Then download the dataset from [here](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz). Move the file txt_sentokens to the src directory.

First run the ```train.py``` .This will train the model and create a file 'my_modelCNN.h5'

Then run the ```predict.py```.This will ask the user to input the review and gives the film review based on that'.


## Built With ❤️ 

* [Python3.6](https://docs.python.org/3.6/) - ⚠️️ Warning : Tensorflow is not supported on any version of python above 3.6 as of now.
* [Tensorflow](https://www.tensorflow.org/api_docs) - The deep learning platform used
* [NLTK](https://www.nltk.org/) - The Natural Language ToolKit used to process the data

## 💁🏻 Contributing

🍴 Fork this repo! and do contribute...

Please feel free to raise any issue...
