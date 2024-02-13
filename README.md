# Movie-Genre-Classifier

## Overview
The Movie Genre Classifier project aims to develop a predictive model capable of automatically assigning genres to movies based on their plot summaries. Leveraging natural language processing (NLP) techniques and machine learning algorithms, the classifier analyzes textual data to predict the genre(s) of a given movie.

## Dataset
The dataset used in this project was obtained from Kaggle and contains movie plot summaries scraped from Wikipedia. It consists of 34,886 entries with the following information:

Release Year: Year of release <br>
Title: Title of the movies <br>
Origin/Ethnicity: Country of origin of the movies <br>
Director: Director names associated with the movies <br>
Cast: Cast names associated with the movies <br>
Genre: Movie genre(s) <br>
Wiki Page: Wikipedia page of the movies <br>
Plot: Plot summary of the movies <br>

## Code Structure
The codebase is organized as follows:

### data_exploration.ipynb:  <br>

Jupyter Notebook containing the data exploration and preprocessing steps. <br>
README.md: Documentation providing an overview of the project, dataset, and code structure. <br>
Images/: Directory containing images generated during data exploration (e.g., histograms, bar plots). <br>
utils.py: Python script containing utility functions used in the project. <br>

## Setup
To run the code, follow these steps:

Install the required libraries by running:
### pip install -r requirements.txt

Download the dataset from Kaggle and place it in the project directory.
Open and run the 
### data_exploration.ipynb 
notebook to explore the dataset, preprocess the data, and build the classification model.

## Dependencies
The project relies on the following Python libraries:

pandas <br>
numpy <br>
matplotlib <br>
seaborn <br>
scikit-learn <br>
nltk <br>
spacy <br>
yellowbrick <br>
dataframe_image <br>
mlxtend <br>
