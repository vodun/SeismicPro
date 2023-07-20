from ..survey import Survey
from ..utils import to_list, get_first_defined


class Statics:
    def __init__(self, survey, source_map, receiver_map, corrected_source_map=None):
        self.source_map = source_map
        self.receiver_map = receiver_map
        self.corrected_source_map = get_first_defined(corrected_source_map, source_map)
        self.survey_list = to_list(survey)
        self.is_single_survey = isinstance(survey, Survey)

        self.source_statics = source_map.index_data.set_index(source_map.index_cols)["Delay"]
        self.receiver_statics = receiver_map.index_data.set_index(receiver_map.index_cols)["Delay"]

    # Statics application

    @staticmethod
    def _apply_to_survey(survey, source_statics, receiver_statics, statics_header="Statics"):
        source_statics = source_statics.copy(deep=False)
        source_statics.rename("_source_delay", inplace=True)
        receiver_statics = receiver_statics.copy(deep=False)
        receiver_statics.rename("_receiver_delay", inplace=True)

        headers = survey.headers
        headers = headers.join(source_statics, on=source_statics.index.names)
        headers = headers.join(receiver_statics, on=receiver_statics.index.names)
        headers[statics_header] = headers["_source_delay"] + headers["_receiver_delay"]
        headers.drop(columns=["_source_delay", "_receiver_delay"], inplace=True)
        if len(headers) != len(survey.headers):
            raise ValueError("duplicates after merge")
        statics_survey = survey.copy(ignore="headers")
        statics_survey.headers = headers
        return statics_survey

    def apply(self, statics_header="Statics"):
        if len(self.survey_list) == 1:
            statics_survey_list = [self._apply_to_survey(self.survey_list[0], self.source_statics,
                                                         self.receiver_statics, statics_header)]
        else:
            statics_survey_list = [self._apply_to_survey(survey, self.source_statics.loc[i],
                                                         self.receiver_statics.loc[i], statics_header)
                                   for i, survey in enumerate(self.survey_list)]

        if self.is_single_survey:
            return statics_survey_list[0]
        return statics_survey_list

    # Statics visualization

    def plot(self, by="shot", corrected=True, interactive=False, sort_by=None, center=True):
        by = by.lower()
        if by in {"source", "shot"}:
            statics_map = self.corrected_source_map if corrected else self.source_map
        elif by in {"receiver", "rec"}:
            statics_map = self.receiver_map
        else:
            raise ValueError("Unknown by")

        if interactive:
            index_cols = statics_map.index_cols if len(self.survey_list) == 1 else statics_map.index_cols[1:]
            survey_list = [sur.reindex(index_cols) for sur in self.survey_list]

            def get_gather(index):
                if len(survey_list) == 1:
                    part = 0
                else:
                    part = index[0]
                    index = index[1:]
                survey = survey_list[part]
                gather = survey.get_gather(index, copy_headers=True)
                if sort_by is not None:
                    gather = gather.sort(by=sort_by)
                return gather

            def plot_gather(ax, coords, index, **kwargs):
                _ = coords, kwargs
                gather = get_gather(index)
                gather.plot(ax=ax, title="Gather without statics corrections applied")

            def plot_gather_statics(ax, coords, index, **kwargs):
                _ = coords, kwargs
                gather = get_gather(index)

                source_statics = self.source_statics if len(survey_list) == 1 else self.source_statics.loc[index[0]]
                source_statics = source_statics.copy(deep=False)
                source_statics.rename("_source_delay", inplace=True)
                receiver_statics = self.receiver_statics if len(survey_list) == 1 else self.receiver_statics.loc[index[0]]
                receiver_statics = receiver_statics.copy(deep=False)
                receiver_statics.rename("_receiver_delay", inplace=True)

                headers = gather.headers
                headers = headers.join(source_statics, on=source_statics.index.names)
                headers = headers.join(receiver_statics, on=receiver_statics.index.names)
                headers["_statics"] = headers["_source_delay"] + headers["_receiver_delay"]
                if center:
                    headers["_statics"] = headers["_statics"] - headers["_statics"].mean()
                if len(headers) != len(gather.headers):
                    raise ValueError("duplicates after merge")
                gather = gather.copy(ignore="headers")
                gather.headers = headers
                gather = gather.apply_statics("_statics")
                gather.plot(ax=ax, title="Gather with statics corrections applied")

            plot_on_click = [plot_gather, plot_gather_statics]
        else:
            plot_on_click = None
        statics_map.plot(interactive=interactive, plot_on_click=plot_on_click)
