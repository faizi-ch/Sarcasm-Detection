# Sarcasm Detection

This repository contains code and resources for detecting sarcasm in textual data. The sarcasm detection model is implemented in a Jupyter notebook `sarcasm_detection.ipynb`.

## Dataset

The sarcasm detection model in this repository relies on the following dataset:

**Dataset Source:** [NLPrinceton/SARC](https://github.com/NLPrinceton/SARC)

Please download the dataset from the provided source and follow the instructions in the notebook to load and preprocess the data for training the model.

## Feature Engineering

To enhance the performance of the sarcasm detection models, this repository explores the use of different types of feature embeddings, including TF-IDF and GloVe embeddings. 

For GloVe embeddings:
1. Visit the [GloVe project page](https://nlp.stanford.edu/projects/glove/) for pre-trained word vectors.
2. Download the GloVe embeddings.
3. Once downloaded, extract the contents of the ZIP file in the root directory.

By following these steps, you will have the necessary dataset and GloVe embeddings set up for running the sarcasm detection model.

## Notebook

The main component of this repository is the Jupyter notebook `sarcasm_detection.ipynb`. The notebook provides a comprehensive analysis of sarcasm detection using different feature embeddings (such as TF-IDF and GloVe) and various models.

Please refer to the notebook for detailed instructions on running the code, modifying parameters, and interpreting the results.
