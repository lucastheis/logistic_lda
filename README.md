Logistic LDA
============

This package provides basic implementations of _logistic latent Dirichlet allocation_. It can be
used to discover topics in data containing groups of thematically related items, using either
labeled data or unlabeled data.

If you want to reproduce experiments of our paper, start here instead instead: [:octocat: logistic-lda/experiments](https://github.com/lucastheis/logistic_lda/tree/experiments)

Requirements
------------

* tensorflow == 1.13.2
* numpy >= 1.16.4

The code was tested with the versions above, but older versions might also work.


Getting started
---------------

To get started, download a version of the 20-Newsgroups dataset in TFRecord format:

	./scripts/download_news20.sh

Once downloaded, training can be started with:

	./scripts/train_news20.sh

To use your own dataset, take a look at `./logistic_lda/data.py` for a description of the data
format expected by the training script. Alternatively, modify the training script to use datasets
not stored as TFRecords.

After training has finished, compute predictions on another dataset and evaluate accuracy:

	./scripts/evaluate_news20.sh

The results of the evaluation can be found in `./models/news20/`.


Reference
---------

I. Korshunova, H. Xiong, M. Fedoryszak, L. Theis  
*Discriminative Topic Modeling with Logistic LDA*  
Advances in Neural Information Processing Systems 33, 2019
