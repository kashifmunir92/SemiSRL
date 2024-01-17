<h2 align="center"> <a href="https://github.com/kashifmunir92/semisupervisedsrl">Semi-supervised Semantic Role Labeling with Bidirectional Language Models</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>

<h5 align="center">

[![Paper](https://img.shields.io/badge/Webpage-blue)](https://dl.acm.org/doi/abs/10.1145/3587160)



</h5>
This repository implements the semantic role labeler described in the paper [Semi-supervised Semantic Role Labeling with Bidirectional Language Models](https://dl.acm.org/doi/abs/10.1145/3587160)

The codes are developed based on the [Dynet implementation of biaffine dependency parser](https://github.com/jcyk/Dynet-Biaffine-dependency-parser).


## Usage 

### Data Preprocess
```
python LanguageModel.py
```
### Data Preprocess

```
python preprocess-conll09.py --train /path/to/train.dataset --test /path/to/test.dataset --dev /path/to/dev.dataset
or
python preprocess-conll08.py --train /path/to/train.dataset --test /path/to/test.dataset --dev /path/to/dev.dataset
```
### Train
We use embedding pre-trained by [GloVe](https://nlp.stanford.edu/projects/glove/) (Wikipedia 2014 + Gigaword 5, 6B tokens, 100d)

```
  cd run
  python train.py --config_file ../config.cfg [--dynet-gpu]
```

### Test
```
  cd run
  python test.py --config_file ../config.cfg [--dynet-gpu]
```

All configuration options (see in `run/config.py`) can be specified by the configuration file `config.cfg`.

## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@article{munir2023semi,
  title={Semi-Supervised Semantic Role Labeling with Bidirectional Language Models},
  author={Munir, Kashif and Zhao, Hai and Li, Zuchao},
  journal={ACM Transactions on Asian and Low-Resource Language Information Processing},
  year={2023},
  publisher={ACM New York, NY}
}
```