<div align="center">

![SeismicPro](https://user-images.githubusercontent.com/26159964/196654661-3ff89a60-c17e-47a5-862f-7f6b814a0df9.png)

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#tutorials">Tutorials</a> •
  <a href="#citing-seismicpro">Citation</a>
</p>

[![License](https://img.shields.io/github/license/analysiscenter/batchflow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8-orange.svg)](https://pytorch.org)
[![Status](https://github.com/gazprom-neft/SeismicPro/actions/workflows/status.yml/badge.svg?branch=master&event=push)](https://github.com/gazprom-neft/SeismicPro/actions/workflows/status.yml)
[![Test installation](https://github.com/gazprom-neft/SeismicPro/actions/workflows/test-install.yml/badge.svg?branch=master&event=push)](https://github.com/gazprom-neft/SeismicPro/actions/workflows/test-install.yml)

</div>

---

`SeismicPro` is a framework for acceleration of pre-stack seismic data processing with deep learning models.

Main features:
* Read pre-stack data in `SEG-Y` format at any exploration stage in a highly efficient manner
* Load and utilize stacking velocities, times of first breaks, and other types of auxiliary data from multiple geological frameworks
* Transform seismic data with both general and complex task-specific methods in a massively parallel way
* Combine processing routines into concise and readable pipelines
* Solve applied tasks with a wide range of neural network architectures from a vanilla `UNet` to sophisticated `EfficientNet`s defined in just a few lines of code
* Evaluate the obtained results using interactive quality maps

## Installation

> `SeismicPro` module is in the beta stage. Your suggestions and improvements via [issues](https://github.com/gazprom-neft/SeismicPro/issues) are very welcome.

> Note that the [Benchmark](./benchmark/) module may not work on Windows due to dependency issues. Use it with caution.

`SeismicPro` is compatible with Python 3.8+ and is tested on Ubuntu 20.04 and Windows Server 2022.

* Installation as a python package using [pip](https://pip.pypa.io/en/stable/):
    ```bash
    pip3 install git+https://github.com/gazprom-neft/SeismicPro.git
    ```
* Installation as a python package using [pipenv](https://docs.pipenv.org/):
    ```bash
    pipenv install git+https://github.com/gazprom-neft/SeismicPro.git#egg=SeismicPro
    ```
* Cloning the project repository:
    ```bash
    git clone https://github.com/gazprom-neft/SeismicPro.git
    ```

## Getting Started

`SeismicPro` provides a simple interface to work with pre-stack data.

```python
import seismicpro as spr
```

A single `SEG-Y` file can be represented by a `Survey` instance that stores a requested subset of trace headers and allows for gather loading:

```python
survey = spr.Survey(path_to_file, header_index='FieldRecord', header_cols='offset')
```

`header_index` argument specifies how individual traces are combined into gathers: in this example, we consider common source gathers. Both `header_index` and `header_cols` correspond to names of trace headers in [segyio](https://segyio.readthedocs.io/en/latest/segyio.html#trace-header-keys).

All loaded headers are stored in `headers` attribute as a `pd.DataFrame` indexed by passed `header_index`:

```python
survey.headers.head()
```

|                 |   offset |   TRACE_SEQUENCE_FILE |
|----------------:|---------:|----------------------:|
| **FieldRecord** |          |                       |
|         **175** |      326 |                     1 |
|         **175** |      326 |                     2 |
|         **175** |      333 |                     3 |
|         **175** |      334 |                     4 |
|         **175** |      348 |                     5 |

A randomly selected gather can be obtained by calling `sample_gather` method:

```python
gather = survey.sample_gather()
```

Let's take a look at it being sorted by offset:

```python
gather.sort(by='offset').plot()
```

![gather](https://user-images.githubusercontent.com/26159964/196198315-00ac9178-2a14-4e01-b493-77eed8eed144.png)

Moreover, processing methods can be combined into compact pipelines like the one below which performs automatic stacking velocity picking and gather stacking:

```python
stacking_pipeline = (dataset
    .pipeline()
    .load(src="raw")
    .sort(src="raw", by="offset")
    .mute(src="raw", dst="muted_raw", muter=muter)
    .calculate_vertical_velocity_spectrum(src="muted_raw", dst="raw_semb",  win_size=16,
                                          velocities=SEMBLANCE_VELOCITY_RANGE)
    .calculate_stacking_velocity(src="raw_semb", dst="velocity",
                                 start_velocity_range=START_VELOCITY_RANGE,
                                 end_velocity_range=END_VELOCITY_RANGE,
                                 n_times=N_TIMES, n_velocities=N_VELOCITIES)
    .get_central_gather(src="raw")
    .apply_nmo(src="raw", stacking_velocity="velocity")
    .mute(src="raw", muter=muter)
    .stack(src="raw")
    .dump(src="raw", path=STACK_TRACE_PATH)
)

stacking_pipeline.run(BATCH_SIZE, n_epochs=1)
```

## Tutorials
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
