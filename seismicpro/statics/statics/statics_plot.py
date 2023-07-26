from ...utils import PairedPlot, DropdownViewPlot


class StaticsPlot(PairedPlot):
    def __init__(self, statics, by, center=True, sort_by=None, gather_plot_kwargs=None, orientation="horizontal",
                 **kwargs):
        by = by.lower()
        if by in {"source", "shot"}:
            metric_maps_list = [statics.source_statics_map, statics.corrected_source_statics_map,
                                statics.source_elevation_map]
            titles_list = ["Map of source statics", "Map of uphole-corrected source statics", "Surface elevation map"]
        elif by in {"receiver", "rec"}:
            metric_maps_list = [statics.receiver_statics_map, statics.receiver_elevation_map]
            titles_list = ["Map of receiver statics", "Surface elevation map"]
        else:
            raise ValueError("Unknown by")
        index_cols = metric_maps_list[0].index_cols
        if statics.n_surveys > 1:
            index_cols = index_cols[1:]

        self.statics = statics
        self.survey_list = [survey.reindex(index_cols) for survey in statics.survey_list]

        metric_map_plot_kwargs = {
            "plot_on_click": [self.plot_gather, self.plot_gather_statics],
            "plot_on_click_kwargs": gather_plot_kwargs,
            "orientation": orientation,
            **kwargs
        }
        self.metric_maps_list = [mmap.interactive_map_class(mmap, title=title, **metric_map_plot_kwargs)
                                 for mmap, title in zip(metric_maps_list, titles_list)]

        self.center = center
        self.sort_by = sort_by
        super().__init__(orientation=orientation)

    def construct_main_plot(self):
        plot_fn_list = [mmap.main.plot_fn for mmap in self.metric_maps_list]
        title_list = [mmap.main.title for mmap in self.metric_maps_list]
        click_fn = self.metric_maps_list[0].main.click_fn
        return DropdownViewPlot(plot_fn=plot_fn_list, click_fn=click_fn, title=title_list,
                                preserve_clicks_on_view_change=True)

    def construct_aux_plot(self):
        return self.metric_maps_list[0].aux

    def get_gather(self, index):
        if self.statics.n_surveys == 1:
            part = 0
        else:
            part = index[0]
            index = index[1:]
        gather = self.survey_list[part].get_gather(index, copy_headers=True)
        if self.sort_by is not None:
            gather = gather.sort(by=self.sort_by)
        return gather

    def plot_gather(self, ax, coords, index, **kwargs):
        _ = coords
        gather = self.get_gather(index)
        gather.plot(ax=ax, title="Gather without static corrections applied", **kwargs)

    def plot_gather_statics(self, ax, coords, index, **kwargs):
        _ = coords

        if self.statics.n_surveys == 1:
            source_statics = self.statics.source_statics_list[0]
            receiver_statics = self.statics.receiver_statics_list[0]
        else:
            source_statics = self.statics.source_statics_list[index[0]]
            receiver_statics = self.statics.receiver_statics_list[index[0]]
        gather = self.get_gather(index)
        gather = self.statics._apply_to_container(gather, source_statics, receiver_statics, statics_header="_Statics")
        if self.center:
            gather["_Statics"] = gather["_Statics"] - gather["_Statics"].mean()
        gather = gather.apply_statics("_Statics")
        gather.plot(ax=ax, title="Gather with static corrections applied", **kwargs)

    def plot(self):
        super().plot()
        is_lower_better = self.metric_maps_list[0].is_lower_better
        init_click_coords = self.metric_maps_list[0].metric_map.get_worst_coords(is_lower_better)
        self.main.click(init_click_coords)
