from textwrap import dedent

import numpy as np

from .batch import SeismicBatch
from .index import SeismicIndex
from ..batchflow import Dataset


class SeismicDataset(Dataset):
    def __init__(self, index=None, surveys=None, mode=None, batch_class=SeismicBatch, **kwargs):
        if index is None:
            index = SeismicIndex(surveys=surveys, mode=mode, **kwargs)
        super().__init__(index, batch_class=batch_class, **kwargs)

    def __str__(self):
        msg = dedent(f"""
        Dataset index:             {self.index.__class__}
        Batch class:               {self.batch_class}

        """)
        if isinstance(self.index, SeismicIndex):
            msg += self.index._get_index_info(indents='', prefix='dataset.index')
            for survey_name, survey_list in self.index.surveys_dict.items():
                for concat_id, survey in enumerate(survey_list):
                    msg += f"\n{'_'*79}\nSurvey named '{survey_name}' with CONCAT_ID {concat_id}.\n" + str(survey)
        else:
            msg += str(self.index)
        return msg

    def info(self):
        print(self)

    def create_subset(self, index):
        if isinstance(index, SeismicIndex) and isinstance(self.index, SeismicIndex):
            if not index.indices.isin(self.indices).all():
                raise IndexError("Some indices from given index are not present in current SeismicIndex")
            return type(self).from_dataset(self, self.index.create_subset(index, loc_headers=True))
        return super().create_subset(index)

    def collect_stats(self, n_samples=100000, quantile_precision=2, bar=True):
        concat_ids = self.indices.get_level_values(0)
        indices = self.indices.droplevel(0)
        for concat_id in np.unique(concat_ids):
            concat_id_indices = indices[concat_ids == concat_id]
            for survey_list in self.index.surveys_dict.values():
                survey_list[concat_id].collect_stats(indices=concat_id_indices, n_samples=n_samples,
                                                     quantile_precision=quantile_precision, bar=bar)
