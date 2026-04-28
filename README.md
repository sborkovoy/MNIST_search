# MNIST/EMNIST browser

This project implements an image similarity search pipeline on MNIST/EMNIST using neural network embeddings.

Pipeline:

1. Preprocess input image (normalize, scale, resize).
2. Encode the image into an embedding using a pretrained model.
3. Find top k nearest neighbors in the dataset.
4. Display the query image with its closest matches and labels.