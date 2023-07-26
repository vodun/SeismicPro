import warnings
from functools import partial

import numpy as np
import pandas as pd
import polars as pl

from .statics_plot import StaticsPlot
from ...survey import Survey
from ...metrics import MetricMap
from ...utils import to_list, align_args


class Statics:
    def __init__(self, survey, source_statics, source_id_cols, receiver_statics, receiver_id_cols, validate=True):
        self.is_single_survey = isinstance(survey, Survey)
        self.survey_list = to_list(survey)

        self.source_id_cols = source_id_cols
        self.source_statics_list = to_list(source_statics)
        if len(self.source_statics_list) != len(self.survey_list):
            raise ValueError

        self.receiver_id_cols = receiver_id_cols
        self.receiver_statics_list = to_list(receiver_statics)
        if len(self.receiver_statics_list) != len(self.survey_list):
            raise ValueError

        if validate:
            self.validate_statics()

        # Construct source maps
        source_statics, source_id_cols = self._concatenate_statics(self.source_statics_list, source_id_cols)
        source_map_cls = partial(MetricMap, coords=source_statics[["SourceX", "SourceY"]],
                                 index=source_statics[source_id_cols])
        self.source_statics_map = source_map_cls(values=source_statics["Statics"])
        self.corrected_source_statics_map = source_map_cls(values=source_statics["SurfaceStatics"])
        self.source_elevation_map = source_map_cls(values=source_statics["SourceSurfaceElevation"])

        # Construct receiver maps
        receiver_statics, receiver_id_cols = self._concatenate_statics(self.receiver_statics_list, receiver_id_cols)
        receiver_map_cls = partial(MetricMap, coords=receiver_statics[["GroupX", "GroupY"]],
                                   index=receiver_statics[receiver_id_cols])
        self.receiver_statics_map = receiver_map_cls(values=receiver_statics["Statics"])
        self.receiver_elevation_map = receiver_map_cls(values=receiver_statics["ReceiverGroupElevation"])

    @property
    def n_surveys(self):
        return len(self.survey_list)

    @staticmethod
    def _concatenate_statics(statics_list, id_cols):
        if len(statics_list) == 1:
            return statics_list[0], id_cols
        statics = pd.concat(statics_list)
        statics["Part"] = np.concatenate([np.full(len(stat), i) for i, stat in enumerate(statics_list)])
        id_cols = ["Part"] + to_list(id_cols)
        return statics, id_cols

    def _validate_survey_statics(self, survey, source_statics, receiver_statics):
        survey_headers_lazy = survey.get_polars_headers().lazy()
        source_statics_lazy = pl.from_pandas(source_statics, rechunk=False).lazy()
        receiver_statics_lazy = pl.from_pandas(receiver_statics, rechunk=False).lazy()

        expr_list = [
            survey_headers_lazy.select(self.source_id_cols).unique(),
            source_statics_lazy.select(self.source_id_cols).unique(),
            survey_headers_lazy.select(self.receiver_id_cols).unique(),
            receiver_statics_lazy.select(self.receiver_id_cols).unique(),
        ]
        unique_sources, unique_statics_sources, unique_receivers, unique_statics_receivers = pl.collect_all(expr_list)

        if len(unique_statics_sources) != len(source_statics):
            raise ValueError("Source statics contain sources with duplicated indices")
        if not unique_sources.join(unique_statics_sources, on=self.source_id_cols, how="anti").is_empty():
            warnings.warn("Source statics miss some sources from the survey. Their statics will be set to 0")

        if len(unique_statics_receivers) != len(receiver_statics):
            raise ValueError("Receiver statics contain receivers with duplicated indices")
        if not unique_receivers.join(unique_statics_receivers, on=self.receiver_id_cols, how="anti").is_empty():
            warnings.warn("Receiver statics miss some receivers from the survey. Their statics will be set to 0")

    def validate_statics(self):
        data_iterator = zip(self.survey_list, self.source_statics_list, self.receiver_statics_list)
        for survey, source_statics, receiver_statics in data_iterator:
            self._validate_survey_statics(survey, source_statics, receiver_statics)

    # Statics application

    def _apply_to_container(self, container, source_statics, receiver_statics, statics_header="Statics"):
        container_headers = container.get_polars_headers()
        loaded_headers = container_headers.columns
        indexed_by = container.indexed_by

        source_id_cols = to_list(self.source_id_cols)
        source_statics = pl.from_pandas(source_statics, rechunk=False)
        source_statics = source_statics.select(*source_id_cols, pl.col("Statics").alias("_SourceStatics"))
        container_headers = container_headers.join(source_statics, how="left", on=source_id_cols)
        if container_headers.select("_SourceStatics").null_count().item():
            container_headers = container_headers.with_columns(pl.col("_SourceStatics").fill_null(0))

        receiver_id_cols = to_list(self.receiver_id_cols)
        receiver_statics = pl.from_pandas(receiver_statics, rechunk=False)
        receiver_statics = receiver_statics.select(*receiver_id_cols, pl.col("Statics").alias("_ReceiverStatics"))
        container_headers = container_headers.join(receiver_statics, how="left", on=receiver_id_cols)
        if container_headers.select("_ReceiverStatics").null_count().item():
            container_headers = container_headers.with_columns(pl.col("_ReceiverStatics").fill_null(0))

        statics_expr = (pl.col("_SourceStatics") + pl.col("_ReceiverStatics")).alias(statics_header)
        headers = container_headers.select(*loaded_headers, statics_expr).to_pandas()
        headers.set_index(indexed_by, inplace=True)

        statics_container = container.copy(ignore="headers")
        statics_container.headers = headers
        return statics_container

    def apply(self, statics_header="Statics"):
        _, statics_header = align_args(self.survey_list, statics_header)
        data_iterator = zip(self.survey_list, self.source_statics_list, self.receiver_statics_list, statics_header)
        statics_survey_list = [self._apply_to_container(survey, source_statics, receiver_statics, header)
                               for survey, source_statics, receiver_statics, header in data_iterator]
        if self.is_single_survey:
            return statics_survey_list[0]
        return statics_survey_list

    # Statics visualization

    def plot(self, by, center=True, sort_by=None, gather_plot_kwargs=None, **kwargs):
        statics_plot = StaticsPlot(self, by=by, center=center, sort_by=sort_by, gather_plot_kwargs=gather_plot_kwargs,
                                   **kwargs)
        statics_plot.plot()
