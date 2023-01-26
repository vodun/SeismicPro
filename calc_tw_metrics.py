

import os
import sys

import argparse

sys.path.insert(0, './batchflow')

from seismicpro import Survey

# from seismicpro.survey.metrics import TraceAbsMean, Std, TraceMaxAbs, MaxClipsLen, MaxConstLen, DeadTrace, WindowRMS

def calc(inp_path, metrics=None):

    header_index = 'FieldRecord'
    USE_COLS = [
        'TraceNumber',
        'FieldRecord',
        'offset',
        'SourceX', 'SourceY',
        'GroupX',  'GroupY',
    ]

    survey = Survey(inp_path, header_index=header_index, header_cols=USE_COLS, name="raw")

    survey.qc_tracewise(metrics=metrics, chunk_size=1000)


    def write_column(survey, name, path):
        col_space=8

        rows = survey[['FieldRecord', 'TraceNumber', name]]

        # SEG-Y specification states that all headers values are integers, but first break values can be float
        row_fmt = '{:{col_space}.0f}' * (rows.shape[1] - 1) + '{:{col_space}.4f}\n'
        fmt = row_fmt * len(rows)
        rows_as_str = fmt.format(*rows.ravel(), col_space=col_space)

        with open(path, 'w', encoding="UTF-8") as f:
            f.write(rows_as_str)

    for metric_name in survey.qc_metrics.keys():
        write_column(survey, metric_name, os.path.splitext(inp_path)[0] + '_' + metric_name)

def parse_args():
    """ Read the model and data paths and run inference pipeline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help="Path to input cube",
                        required=True)
    args = parser.parse_args()

    calc(args.input)

if __name__ == "__main__":
    parse_args()
