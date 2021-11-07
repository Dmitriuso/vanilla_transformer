# Vanilla Transformer

An Attention Is All You Need seq2seq model.

## Development practices

To setup the automated code formatting and lint script run `pip install pre-commit && pre-commit install`.
This will run [`pre-commit`](https://pre-commit.com/) each time you do a commit.
If you want to run `pre-commit` yourself do `pre-commit run -a`.

*NB*: Do not forget about `.pre-commit-config.yaml`.

## Citation

```
@inproceedings{NIPS2017_3f5ee243,
 author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, \L ukasz and Polosukhin, Illia},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Attention is All you Need},
 url = {https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf},
 volume = {30},
 year = {2017}
}
```
