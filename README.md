# Using SGD on MNIST

## About The Data

    Notes from fastai course lesson 9:

The good news is that modern machine learning can be distilled down to a couple of key techniques that are of very wide
applicability. Recent studies have shown that the vast majority of datasets can be best modeled with just two methods:

Ensembles of decision trees (i.e. Random Forests and Gradient Boosting Machines), mainly for structured data (such as
you might find in a database table at most companies). We looked at random forests in depth as we analyzed the Blue 
Book for Bulldozers dataset.

Multi-layered neural networks learnt with SGD (i.e. shallow and/or deep learning), mainly for unstructured data 
(such as audio, vision, and natural language)

In this lesson, we will start on the 2nd approach (a neural network with SGD) by analyzing the MNIST dataset. 
You may be surprised to learn that logistic regression is actually an example of a simple neural net!

In this lesson, we will be working with MNIST, a classic data set of hand-written digits. 
Solutions to this problem are used by banks to automatically recognize the amounts on checks, and by the postal service 
to automatically recognize zip codes on mail.

