<div align="center">

![SeismicPro](https://user-images.githubusercontent.com/19351782/125063408-1bcdab80-e0b8-11eb-96c2-719bc640da36.png)

<p align="center">
  <a href="">Docs</a> •
  <a href="#installation">Installation</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="tutorials">Tutorials</a> •
  <a href="#citing-seismicpro">Citation</a>
</p>

[![License](https://img.shields.io/github/license/analysiscenter/batchflow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8-orange.svg)](https://pytorch.org)
[![Status](https://github.com/gazprom-neft/SeismicPro/workflows/status/badge.svg)](https://github.com/gazprom-neft/SeismicPro/actions?query=workflow%3Astatus)
[![Test installation](https://github.com/gazprom-neft/SeismicPro/actions/workflows/test-install.yml/badge.svg)](https://github.com/gazprom-neft/SeismicPro/actions/workflows/test-install.yml)

</div>

---

`SeismicPro` is a framework for accelerating processing of pre-stack seismic data with deep learning models.

Main features:

* Load pre-stack data in `SEG-Y` format at any exploration stage in a highly efficient manner
* Utilize stacking velocities, times of first breaks, and other types of auxiliary data from multiple geological frameworks
* Transform seismic data by both general and complex task-specific methods in a massively parallel way
* Combine processing functions into concise and readable pipelines
* Define a wide range of neural network architectures from vanilla `UNet` to sophisticated `EfficientNet`s with simple and intuitive configurations in just a few lines of code

## Installation

> `SeismicPro` module is in the beta stage. Your suggestions and improvements via [issues](https://github.com/gazprom-neft/SeismicPro/issues) are very welcome.

`SeismicPro` is compatible with Python 3.8+ and tested on Ubuntu 20.04 and Windows Server 2019.

> Note that the [Benchmark](./seismicpro/benchmark/) module and [Research](./seismicpro/batchflow/batchflow/research/) may not work on Windows due to dependency issues. Use it with caution.

### Installation as a python package

With [pip](https://pip.pypa.io/en/stable/):

    pip3 install git+https://github.com/gazprom-neft/SeismicPro.git

With [pipenv](https://docs.pipenv.org/):

    pipenv install git+https://github.com/gazprom-neft/SeismicPro.git#egg=SeismicPro

### Installation as a project repository

When cloning a repo from GitHub use ``--recursive`` flag to make sure that ``batchflow`` submodule is also cloned.

    git clone --recursive https://github.com/gazprom-neft/SeismicPro.git

## Getting Started

`SeismicPro` provides a simple interface to work with pre-stack data.

```python
import seismicpro
```

A single `SEG-Y` file can be represented by a `Survey` instance that stores a requested subset of trace headers and allows for gather loading:

```python
survey = seismicpro.Survey(path_to_file, header_index='FieldRecord', header_cols='offset')
```

`header_index` argument specifies how individual traces are combined into gathers: in this example, we consider common source gathers. Both `header_index` and `header_cols` correspond to names of trace headers in [segyio](https://segyio.readthedocs.io/en/latest/segyio.html#constants).

All loaded headers are stored in `headers` attribute as a `pd.DataFrame` indexed by passed `header_index`:

```python
survey.headers.head()
```

| **FieldRecord** | **offset** | **TRACE_SEQUENCE_FILE** |
|----------------:|-----------:|------------------------:|
|         **175** |       6455 |                       1 |
|         **175** |       6567 |                       2 |
|         **175** |       6683 |                       3 |
|         **175** |       6805 |                       4 |
|         **175** |       6932 |                       5 |

A randomly selected gather can be obtained by calling `sample_gather` method:

```python
gather = survey.sample_gather()
```

Let's take a look at it being sorted by offset:

```python
gather.sort(by='offset').plot()
```

![gather](https://i.imgur.com/qv0SsEE.png)

Moreover, processing methods can be combined into compact pipelines like the one below which performs automatic stacking velocity picking and gather stacking:

```python
stacking_pipeline = (dataset
    .pipeline()
    .load(src="raw")
    .sort(src="raw", by="offset")
    .mute(src="raw", dst="muted_raw", muter=muter)
    .calculate_semblance(src="muted_raw", dst="raw_semb",
                         velocities=SEMBLANCE_VELOCITY_RANGE, win_size=8)
    .calculate_stacking_velocity(src="raw_semb", dst="velocity",
                                 start_velocity_range=START_VELOCITY_RANGE,
                                 end_velocity_range=END_VELOCITY_RANGE,
                                 n_times=N_TIMES, n_velocities=N_VELOCITIES)
    .get_central_cdp(src="raw")
    .apply_nmo(src="raw", stacking_velocity="velocity")
    .mute(src="raw", muter=muter, fill_value=np.nan)
    .stack(src="raw")
    .dump(src="raw", path=STACK_TRACE_PATH, copy_header=False)
)

stacking_pipeline.run(BATCH_SIZE, n_epochs=1)
```

You can get more familiar with the framework and its functionality by reading [SeismicPro tutorials](tutorials).

## Citing SeismicPro

Please cite `SeismicPro` in your publications if it helps your research.

    Khudorozhkov R., Illarionov E., Broilovskiy A., Kalashnikov N., Podvyaznikov D., Arefina A., Kuvaev A., SeismicPro library for seismic data processing and ML models training and inference. 2019.

```
@misc{seismicpro_2019,
  author       = {R. Khudorozhkov and E. Illarionov and A. Broilovskiy and N. Kalashnikov and D. Podvyaznikov and A. Arefina and A. Kuvaev},
  title        = {SeismicPro library for seismic data processing and ML models training and inference},
  year         = 2019
}
```
