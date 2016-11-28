# Pinterest Multimodal Dataset ToolBox

Created by [Junhua Mao](www.stat.ucla.edu/~junhua.mao)

## Introduction

This is a toolbox to download and manage the released part of the Pinterest40M multimodal dataset introduced in the paper *Training and Evaluating Multimodal Word Embeddings with Large-scale Web Annotated Images* [Paper](https://papers.nips.cc/paper/6590-training-and-evaluating-multimodal-word-embeddings-with-large-scale-web-annotated-images).
More information can be found on the [Project Page](http://www.stat.
ucla.edu/~junhua.mao/multimodal_embedding.html).

## Cite

If you find this dataset or toolbox useful in your research, please cite:

  @inproceedings{mao2016training,
    title={Training and Evaluating Multimodal Word Embeddings with Large-scale Web Annotated Images},
    author={Mao, Junhua and Xu, Jiajing and Jing, Yushi and Yuille, Alan},
    booktitle={NIPS},
    year={2016}
  }

## Toolbox Installation and Data Downloading
### Download and setup meta files.

Suppose that toolkit is install on $PATH_PTool:
  ```Shell
  cd $PATH_PTool
  bash download_meta.sh
  ```
  
### Download images.

You can easily download images in parallel (12 workers by default) and resize the downloaded images to 224x224:
  ```Shell
  cd $PATH_PTool
  python download_images.py
  ```
There are ~5 million images in the dataset. The download process can take days.

The script allows you to resume your downloading at any time.
Just re-run download_images.py if your downloading is shutted down unexpectedly.
It is possible that you failed to access some of the urls at the first time.
Re-run download_images.py to have another try.

You are welcome to read *download_images.py* and *py_utils.py* for personalized and advanced downloading settings (e.g. see the docstring of py_utils.PinDataset.download_images).

## Demo

View *demo.ipynb* for how to use this toolbox.

## Recommended dataset split

Use pin_2016_v1_0000.npy to pin_2016_v1_0097.npy as the training set.
Use pin_2016_v1_0098.npy as the validation set.
Use pin_2016_v1_0099.npy as the test set.

## License

The copyright of the annotations and the images belongs to the original source.
This meta data file can be used for research proposes only.

This toolbox is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
