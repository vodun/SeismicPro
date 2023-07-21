import warnings

import polars as pl

from ..survey import Survey
from ..utils import to_list, align_args


class Statics:
    def __init__(self, survey, source_delays, source_id_cols, receiver_delays, receiver_id_cols, validate=True):
        self.is_single_survey = isinstance(survey, Survey)
        self.survey_list = to_list(survey)

        self.source_id_cols = source_id_cols
        self.source_delays_list = to_list(source_delays)
        if len(self.source_delays_list) != len(self.survey_list):
            raise ValueError

        self.receiver_id_cols = receiver_id_cols
        self.receiver_delays_list = to_list(receiver_delays)
        if len(self.receiver_delays_list) != len(self.survey_list):
            raise ValueError

        if validate:
            self.validate_statics()

    def _validate_survey_statics(self, survey, source_delays, receiver_delays):
        survey_headers_lazy = survey.get_polars_headers().lazy()
        source_delays_lazy = pl.from_pandas(source_delays, rechunk=False).lazy()
        receiver_delays_lazy = pl.from_pandas(receiver_delays, rechunk=False).lazy()

        expr_list = [
            survey_headers_lazy.select(self.source_id_cols).unique(),
            source_delays_lazy.select(self.source_id_cols).unique(),
            survey_headers_lazy.select(self.receiver_id_cols).unique(),
            receiver_delays_lazy.select(self.receiver_id_cols).unique(),
        ]
        unique_sources, unique_delay_sources, unique_receivers, unique_delay_receivers = pl.collect_all(expr_list)

        if len(unique_delay_sources) != len(source_delays):
            raise ValueError("Source statics contain sources with duplicated indices")
        if len(unique_sources.join(unique_delay_sources, on=self.source_id_cols)) != len(unique_sources):
            warnings.warn("Source statics miss some sources from the survey. Their statics will be set to 0")

        if len(unique_delay_receivers) != len(unique_receivers):
            raise ValueError("Receiver statics contain receivers with duplicated indices")
        if len(unique_sources.join(unique_delay_sources, on=self.source_id_cols)) != len(unique_sources):
            warnings.warn("Receiver statics miss some receivers from the survey. Their statics will be set to 0")

    def validate_statics(self):
        data_iterator = zip(self.survey_list, self.source_delays_list, self.receiver_delays_list)
        for survey, source_delays, receiver_delays in data_iterator:
            self._validate_survey_statics(survey, source_delays, receiver_delays)

        # ------------------------------------------------------------------------------

        # self.source_map = source_map
        # self.receiver_map = receiver_map
        # self.corrected_source_map = get_first_defined(corrected_source_map, source_map)
        # self.survey_list = to_list(survey)
        # self.is_single_survey = isinstance(survey, Survey)

        # self.source_statics = source_map.index_data.set_index(source_map.index_cols)["Delay"]
        # self.receiver_statics = receiver_map.index_data.set_index(receiver_map.index_cols)["Delay"]

        # add_part = len(survey_list) > 1
        # source_delays_list = []
        # receiver_delays_list = []
        # for i, survey in enumerate(survey_list):
        #     source_delays = self._get_source_delays(survey, source_id_cols, uphole_correction_method,
        #                                             **estimate_delays_kwargs)
        #     receiver_delays = self._get_receiver_delays(survey, receiver_id_cols, **estimate_delays_kwargs)
        #     if add_part:
        #         source_delays.insert(0, "Part", i)
        #         receiver_delays.insert(0, "Part", i)
        #     source_delays_list.append(source_delays)
        #     receiver_delays_list.append(receiver_delays)
        # source_delays = pd.concat(source_delays_list, ignore_index=True, copy=False)
        # receiver_delays = pd.concat(receiver_delays_list, ignore_index=True, copy=False)
        # source_id_cols = to_list(source_id_cols)
        # if add_part:
        #     source_id_cols = ["Part"] + source_id_cols
        # receiver_id_cols = to_list(receiver_id_cols)
        # if add_part:
        #     receiver_id_cols = ["Part"] + receiver_id_cols

        # source_map = MetricMap(source_delays[["SourceX", "SourceY"]], source_delays["Delay"], index=source_delays[source_id_cols])
        # corrected_source_map = MetricMap(source_delays[["SourceX", "SourceY"]], source_delays["SurfaceDelay"], index=source_delays[source_id_cols])
        # receiver_map = MetricMap(receiver_delays[["GroupX", "GroupY"]], receiver_delays["Delay"], index=receiver_delays[receiver_id_cols])
        # survey_list = survey_list[0] if is_single_survey else survey_list
        # return Statics(survey_list, source_map, receiver_map, corrected_source_map)

    # Statics application

    def _apply_to_container(self, container, source_statics, receiver_statics, statics_header="Statics"):
        container_headers = container.get_polars_headers()
        loaded_headers = container_headers.columns
        indexed_by = container.indexed_by

        source_id_cols = to_list(self.source_id_cols)
        source_statics = pl.from_pandas(source_statics, rechunk=False)
        source_statics = source_statics.select(*source_id_cols, pl.col("Delay").alias("_SourceDelay"))
        container_headers = container_headers.join(source_statics, how="left", on=source_id_cols)
        if container_headers.select("_SourceDelay").null_count().item():
            container_headers = container_headers.with_columns(pl.col("_SourceDelay").fill_null(0))

        receiver_id_cols = to_list(self.receiver_id_cols)
        receiver_statics = pl.from_pandas(receiver_statics, rechunk=False)
        receiver_statics = receiver_statics.select(*receiver_id_cols, pl.col("Delay").alias("_ReceiverDelay"))
        container_headers = container_headers.join(receiver_statics, how="left", on=receiver_id_cols)
        if container_headers.select("_ReceiverDelay").null_count().item():
            container_headers = container_headers.with_columns(pl.col("_ReceiverDelay").fill_null(0))

        delay_expr = (pl.col("_SourceDelay") + pl.col("_ReceiverDelay")).alias(statics_header)
        headers = container_headers.select(*loaded_headers, delay_expr).to_pandas()
        headers.set_index(indexed_by, inplace=True)

        statics_container = container.copy(ignore="headers")
        statics_container.headers = headers
        return statics_container

    def apply(self, statics_header="Statics"):
        _, statics_header = align_args(self.survey_list, statics_header)
        data_iterator = zip(self.survey_list, self.source_delays_list, self.receiver_delays_list, statics_header)
        statics_survey_list = [self._apply_to_container(survey, source_statics, receiver_statics, header)
                               for survey, source_statics, receiver_statics, header in data_iterator]
        if self.is_single_survey:
            return statics_survey_list[0]
        return statics_survey_list

    # Statics visualization

    # def plot(self, by="shot", corrected=True, interactive=False, sort_by=None, center=True):
    #     by = by.lower()
    #     if by in {"source", "shot"}:
    #         statics_map = self.corrected_source_map if corrected else self.source_map
    #     elif by in {"receiver", "rec"}:
    #         statics_map = self.receiver_map
    #     else:
    #         raise ValueError("Unknown by")

    #     if interactive:
    #         index_cols = statics_map.index_cols if len(self.survey_list) == 1 else statics_map.index_cols[1:]
    #         survey_list = [sur.reindex(index_cols) for sur in self.survey_list]

    #         def get_gather(index):
    #             if len(survey_list) == 1:
    #                 part = 0
    #             else:
    #                 part = index[0]
    #                 index = index[1:]
    #             survey = survey_list[part]
    #             gather = survey.get_gather(index, copy_headers=True)
    #             if sort_by is not None:
    #                 gather = gather.sort(by=sort_by)
    #             return gather

    #         def plot_gather(ax, coords, index, **kwargs):
    #             _ = coords, kwargs
    #             gather = get_gather(index)
    #             gather.plot(ax=ax, title="Gather without statics corrections applied")

    #         def plot_gather_statics(ax, coords, index, **kwargs):
    #             _ = coords, kwargs
    #             gather = get_gather(index)

    #             source_statics = self.source_statics if len(survey_list) == 1 else self.source_statics.loc[index[0]]
    #             source_statics = source_statics.copy(deep=False)
    #             source_statics.rename("_source_delay", inplace=True)
    #             receiver_statics = self.receiver_statics if len(survey_list) == 1 else self.receiver_statics.loc[index[0]]
    #             receiver_statics = receiver_statics.copy(deep=False)
    #             receiver_statics.rename("_receiver_delay", inplace=True)

    #             headers = gather.headers
    #             headers = headers.join(source_statics, on=source_statics.index.names)
    #             headers = headers.join(receiver_statics, on=receiver_statics.index.names)
    #             headers["_statics"] = headers["_source_delay"] + headers["_receiver_delay"]
    #             if center:
    #                 headers["_statics"] = headers["_statics"] - headers["_statics"].mean()
    #             if len(headers) != len(gather.headers):
    #                 raise ValueError("duplicates after merge")
    #             gather = gather.copy(ignore="headers")
    #             gather.headers = headers
    #             gather = gather.apply_statics("_statics")
    #             gather.plot(ax=ax, title="Gather with statics corrections applied")

    #         plot_on_click = [plot_gather, plot_gather_statics]
    #     else:
    #         plot_on_click = None
    #     statics_map.plot(interactive=interactive, plot_on_click=plot_on_click)
