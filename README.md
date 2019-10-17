# Logistic LDA

This package provides implementations of _logistic latent Dirichlet allocation_. Logistic LDA can be used to discover topics in data containing groups of
thematically related items, using either labeled data or unlabeled data.

Use code on this branch to reproduce experiments of our paper or if you want to experiment with empirical risk minimization for logistic LDA.
For a more minimal implementation of variational inference only, see: [:octocat: logistic-lda/master](https://github.com/lucastheis/logistic_lda/tree/master)

### Requirements

The code was tested with the following settings:

- python3
- tensorflow==1.8

Extra libraries needed for the Pinterest experiments:

- pymongo==3.9.0
- Pillow==4.3.0
- tensorflow-hub==0.3.0 (make sure you also have a compatible version of protobuf)


### Datasets

We list the datasets that should be stored in a `data` directory inside a project folder.

**Newsgroups 20**

Download our version of the 20-Newsgroups dataset:

```
bash scripts/download_news20.sh
```


**Pinterest**

Download Pinterest board/pin IDs and their categories (Geng et al., ICCV'15):

```
bash scripts/download_pinterest_metadata.sh
```

Reorganize the categories, download Pinterest images and write everything into TFRecords:

```
python scripts/download_pinterest_images.py
```



### Training and testing

The makefile includes all the commands to train and evaluate models from the paper.
Also, see the makefile for all possible parameters.

For example, to train and evaluate the document classification logistic LDA model that uses a cross-entropy loss:

```
make ng20_supervised_ce
```
This will create a model directory `metadata/ng20_supervised_ce` with tensorflow checkpoints, 
model parameters in `args.json`, classification results in txt files and test set predictions in a csv file.   

To run multiple experiments with hyperparameters sampled randomly from certain values: 
```
for i in {1..30};  do make search_ng_20_supervised; done
```
To view the results of these models, run the script:
```
python scripts/collect_results.py
```



### Citation

```
@incollection{loglda2019,
  title = {Discriminative Topic Modeling with Logistic LDA},
  author = {Korshunova, Iryna and Xiong, Hanchen and Fedoryszak, Mateusz and Theis, Lucas},
  booktitle = {Advances in Neural Information Processing Systems 33},
  year = {2019}
}
```
