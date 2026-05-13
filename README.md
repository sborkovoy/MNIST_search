# MNIST/EMNIST browser

This project implements an image similarity search pipeline on MNIST/EMNIST using neural network embeddings.

Pipeline:

1. Preprocess input image (normalize, scale, resize).
2. Encode the image into an embedding using a pretrained model.
3. Find top k nearest neighbors in the dataset.
4. Display the query image with its closest matches and labels.

---
For Hadoop:

1. Generate typed bytes file of the data directory:
```
/nn/bin/python gen_tb.py > all.tb
```
2. Then run process
```
hadoop-streaming-raw --input /BespaytyyIV/project/all.tb --output /BespyatyyIV/output_project --mapper "/nn/bin/python mapper.py"
```
