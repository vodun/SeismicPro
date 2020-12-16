import numpy as np

from seismicpro import FieldIndex, SeismicBatch, SeismicDataset, CustomIndex
from seismicpro.batchflow import action, inbatch_parallel


class GeomFieldIndex(FieldIndex):

    def sort(self, by='offset'):
        df = self.get_df(reset=False)
        df.sort_values([self.name, by], inplace=True)
        self._idf = df

    def keep_first(self, slice):
        df = self.get_df(reset=False)
        df = df.groupby(level=0)
        df = df.apply(lambda _df: _df.iloc[slice])
        df.index = df.index.droplevel()
        self._idf = df

class GeomCustomIndex(CustomIndex):

    def sort(self, by='offset'):
        df = self.get_df(reset=False)
        df.sort_values([self.name, by], inplace=True)
        self._idf = df

    def keep_first(self, slice):
        df = self.get_df(reset=False)
        df = df.groupby(level=0)
        df = df.apply( lambda _df : _df.iloc[slice])
        df.index = df.index.droplevel()
        self._idf = df


class GeomSeismicBatch(SeismicBatch):

    @action
    @inbatch_parallel(init="_init_component", target="f")
    def break_geometry(self, ix, src='raw', dst=None, x=None, y=None, p=0.5,
                       use_clean=True, src_offset='offset', src_gx='GroupX',
                       src_gy='GroupY', src_sx='SourceX', src_sy='SourceY'):
        pos = self.get_pos(None, None, ix)
        raw = getattr(self, src)[pos]
        n = len(raw)
        offset = getattr(self, src_offset)[pos][:n].reshape(1, -1)
        labels = np.array([0, 0]).reshape(1, -1)
        if np.random.binomial(1, p):
            gx = getattr(self, src_gx)[pos][:n]
            gy = getattr(self, src_gy)[pos][:n]
            sx = getattr(self, src_sx)[pos][:n]
            sy = getattr(self, src_sy)[pos][:n]
            calc_offset = np.sqrt((gx-sx) ** 2 + (gy-sy) ** 2)
            if not (np.abs(calc_offset - offset) < 5).all():
                print('offset in SEGY do not match reciever/source coordinates')
            bad_gx, bad_gy = gx + np.reshape(x, (-1, 1)), gy + np.reshape(y, (-1, 1))
            break_offset = np.sqrt((bad_gx - sx) ** 2 + (bad_gy - sy) ** 2)
            break_labels = np.squeeze(np.dstack((x, y)))
            if use_clean:
                offset = np.vstack((offset, break_offset))
                labels = np.vstack((labels, break_labels))
            else:
                offset = break_offset
                labels = break_labels
        order = np.argsort(offset)
        getattr(self, dst[0])[pos] = raw[order]
        getattr(self, dst[1])[pos] = labels
        getattr(self, dst[2])[pos] = offset
        return self

    @action
    @inbatch_parallel(init="_init_component", target="f")
    def LMO(self, ix, V='V', length=100, src_traces='raw', src_offset='offset', dst=None, pad=0):
        pos = self.index.get_pos(ix)
        raw = getattr(self, src_traces)[pos]
        offset = getattr(self, src_offset)[pos]
        if isinstance(V, (int, float)):
            V = np.tile(V, np.max(offset).astype(int) + 1)
        else:
            V = getattr(self, V).loc[ix].values[0]

        n_items = False
        if raw.ndim == 3:
            n_items = len(raw)
            raw = np.concatenate(raw)
            offset = np.concatenate(offset)

        rate = np.diff(self.meta[src_traces]['samples'])[0].astype(int)
        start_sample = [(off / V[int(off)]) / rate  for off in offset]
        start_sample = np.round(start_sample).astype(int)
        trace_samples = [np.arange(start - pad, start + length - pad) for start in start_sample]
        raw_lmo = raw[np.arange(len(raw)).reshape(-1, 1), trace_samples]

        if not n_items:
            getattr(self, dst)[pos] = raw_lmo
        else:
            getattr(self, dst)[pos] = np.array(np.split(raw_lmo, n_items))
        return self

class GeomSeismicDataset(SeismicDataset):

    def __init__(self, index, batch_class=GeomSeismicBatch, preloaded=None, *args, **kwargs):
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, *args, **kwargs)
