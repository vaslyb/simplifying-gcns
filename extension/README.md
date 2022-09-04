In file *arxiv.py* the code for the experiment conducted in *ogbn-arxiv* dataset is given.

For running an experiment with *GCN* model,execute the following command:
```
$ python arxiv.py --model GCN
```

In file *fakenews.py* there is the code for the graph classification tasks.
For running an experiment in "Politifact" with *GAT* model and using features coming from BERT, execute the following command:
```
$ python fakenews.py --model GAT --dataset politifact --feature bert
```

In file *many.py* there is the code for experiments concerning social networks and image classification.

For running an experiment in *Reddit* dataset with *SGC* model, execute the following command:
```
$ python many.py --dataset reddit --model SGC
```
