<div align="center">

**SeismicPro**

`SeismicPro` is a framework for accelerating research and processing of pre-stack seismic data with deep learning models.

<p align="center">
  <a href="">Docs</a> •
  <a href="#installation">Installation</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="tutorials">Tutorials</a> •
  <a href="#citing-seismicpro">Citation</a>
</p>

[![License](https://img.shields.io/github/license/analysiscenter/batchflow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7-orange.svg)](https://pytorch.org)
[![Status](https://github.com/gazprom-neft/SeismicPro/workflows/status/badge.svg)](https://github.com/gazprom-neft/SeismicPro/actions?query=workflow%3Astatus)

</div>

---

`SeismicPro` is a framework for seismic processing that works with pre-stack data and allows for accelerating geological reseach using deep learning models in a massively parallel way.

Main features:

* Process a pre-stack data in `SEG-Y` format and provide an incredible speed of loading seismic traces
* Allows reading various data formats from geological frameworks such as vertical velocities, first break points and others
* Contains massive amount of processing functions that works in parallel
* Combine processing functions in compact pipelines
* Define sophisticated neural networks like `EfficientNet` with simple and intuitive configurations in a few lines of code


## Installation

> `SeismicPro` module is in the beta stage. Your suggestions and improvements are very welcome via [issues](https://github.com/gazprom-neft/SeismicPro/issues).

### Installation as a python package

With [pip](https://pip.pypa.io/en/stable/):

    pip3 install git+https://github.com/gazprom-neft/SeismicPro.git

With [pipenv](https://docs.pipenv.org/):

    pipenv install git+https://github.com/gazprom-neft/SeismicPro.git#egg=SeismicPro

After that just import `seismicpro`:
```python
import seismicpro
```

### Installation as a project repository

When cloning repo from GitHub use flag ``--recursive`` to make sure that ``batchflow`` submodule is also cloned.

    git clone --recursive https://github.com/gazprom-neft/SeismicPro.git


## Getting Started

`SeismicPro` provides a simple interface to process pre-stack data.

To start use framework just write:

```python
import seismicpro
```

Use `Survey` to describe your field headers:

```python
survey = Survey(path_to_file, header_index='FieldRecord', header_cols='offset')
```
`header_index` and `header_cols` corresponds to headers name in [segyio](https://segyio.readthedocs.io/en/latest/segyio.html#constants).

All loaded headers are saved as pd.DataFrame and stored in attribute `headers`:

```python
survey.headers
```

|   FieldRecord |   offset |   TRACE_SEQUENCE_FILE |
|--------------:|---------:|----------------------:|
|           175 |     6455 |                     1 |
|           175 |     6567 |                     2 |
|           175 |     6683 |                     3 |
|           175 |     6805 |                     4 |
|           175 |     6932 |                     5 |

Sample single seismogram use `sample_gather` function.

```python
gather = survey.sample_gather()
```
Apply sorting action to sampled gather and plot result.

```python
gather.sort(by='offset').plot()
```
![gather](https://i.imgur.com/qv0SsEE.png)

Moreover, one can construct any preprocessing pipeline, for example here is a pipeline for calculating stacking velocity:

```python
VELOCITY PIPELINE
```

For more examples, see the [SeismicPro tutorials](tutorials).


## Citing SeismicPro

Please cite SeismicPro in your publications if it helps your research.

    Khudorozhkov R., Illarionov E., Broilovskiy A., Kalashnikov N., Podvyaznikov D., Arefina A., Kuvaev A., SeismicPro library for seismic data processing and ML models training and inference. 2019.

```
@misc{seismicpro_2019,
  author       = {R. Khudorozhkov and E. Illarionov and A. Broilovskiy and N. Kalashnikov and D. Podvyaznikov and A. Arefina and A. Kuvaev},
  title        = {SeismicPro library for seismic data processing and ML models training and inference},
  year         = 2019
}
```
