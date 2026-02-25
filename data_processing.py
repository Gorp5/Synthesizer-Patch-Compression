from dataclasses import dataclass
from typing import Optional
import torch
import pandas as pd

@dataclass
class PatchParam:
    name: str
    p_type: str  # 'm', 'b', 'c', 'x'
    p_min: int
    p_max: int
    n_classes: Optional[int] = None  # for categorical

def get_params():
    p_type = ["m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "c", "c", "m", "m", "m", "m", "b", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "c", "c", "m", "m", "m", "m", "b", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "c", "c", "m", "m", "m", "m", "b", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "c", "c", "m", "m", "m", "m", "b", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "c", "c", "m", "m", "m", "m", "b", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "c", "c", "m", "m", "m", "m", "b", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "x", "m", "b", "m", "m", "m", "m", "b", "c", "m", "m"]
    p_name = ["operator6rate1", "operator6rate2", "operator6rate3", "operator6rate4", "operator6level1", "operator6level2", "operator6level3", "operator6level4", "operator6keyboardlevelscalingbreakpoint", "operator6keyboardlevelscalingleftdepth", "operator6keyboardlevelscalingrightdepth", "operator6keyboardlevelscalingleftcurve", "operator6keyboardlevelscalingrightcurve", "operator6keyboardratescaling", "operator6amplitudemodulationsensitivity", "operator6keyvelocitysensitivity", "operator6operatoroutputlevel", "operator6oscillatormode", "operator6frequencycoarse", "operator6frequencyfine", "operator6frequencydetune", "operator5rate1", "operator5rate2", "operator5rate3", "operator5rate4", "operator5level1", "operator5level2", "operator5level3", "operator5level4", "operator5keyboardlevelscalingbreakpoint", "operator5keyboardlevelscalingleftdepth", "operator5keyboardlevelscalingrightdepth", "operator5keyboardlevelscalingleftcurve", "operator5keyboardlevelscalingrightcurve", "operator5keyboardratescaling", "operator5amplitudemodulationsensitivity", "operator5keyvelocitysensitivity", "operator5operatoroutputlevel", "operator5oscillatormode", "operator5frequencycoarse", "operator5frequencyfine", "operator5frequencydetune", "operator4rate1", "operator4rate2", "operator4rate3", "operator4rate4", "operator4level1", "operator4level2", "operator4level3", "operator4level4", "operator4keyboardlevelscalingbreakpoint", "operator4keyboardlevelscalingleftdepth", "operator4keyboardlevelscalingrightdepth", "operator4keyboardlevelscalingleftcurve", "operator4keyboardlevelscalingrightcurve", "operator4keyboardratescaling", "operator4amplitudemodulationsensitivity", "operator4keyvelocitysensitivity", "operator4operatoroutputlevel", "operator4oscillatormode", "operator4frequencycoarse", "operator4frequencyfine", "operator4frequencydetune", "operator3rate1", "operator3rate2", "operator3rate3", "operator3rate4", "operator3level1", "operator3level2", "operator3level3", "operator3level4", "operator3keyboardlevelscalingbreakpoint", "operator3keyboardlevelscalingleftdepth", "operator3keyboardlevelscalingrightdepth", "operator3keyboardlevelscalingleftcurve", "operator3keyboardlevelscalingrightcurve", "operator3keyboardratescaling", "operator3amplitudemodulationsensitivity", "operator3keyvelocitysensitivity", "operator3operatoroutputlevel", "operator3oscillatormode", "operator3frequencycoarse", "operator3frequencyfine", "operator3frequencydetune", "operator2rate1", "operator2rate2", "operator2rate3", "operator2rate4", "operator2level1", "operator2level2", "operator2level3", "operator2level4", "operator2keyboardlevelscalingbreakpoint", "operator2keyboardlevelscalingleftdepth", "operator2keyboardlevelscalingrightdepth", "operator2keyboardlevelscalingleftcurve", "operator2keyboardlevelscalingrightcurve", "operator2keyboardratescaling", "operator2amplitudemodulationsensitivity", "operator2keyvelocitysensitivity", "operator2operatoroutputlevel", "operator2oscillatormode", "operator2frequencycoarse", "operator2frequencyfine", "operator2frequencydetune", "operator1rate1", "operator1rate2", "operator1rate3", "operator1rate4", "operator1level1", "operator1level2", "operator1level3", "operator1level4", "operator1keyboardlevelscalingbreakpoint", "operator1keyboardlevelscalingleftdepth", "operator1keyboardlevelscalingrightdepth", "operator1keyboardlevelscalingleftcurve", "operator1keyboardlevelscalingrightcurve", "operator1keyboardratescaling", "operator1amplitudemodulationsensitivity", "operator1keyvelocitysensitivity", "operator1operatoroutputlevel", "operator1oscillatormode", "operator1frequencycoarse", "operator1frequencyfine", "operator1frequencydetune", "pitchegrate1", "pitchegrate2", "pitchegrate3", "pitchegrate4", "pitcheglevel1", "pitcheglevel2", "pitcheglevel3", "pitcheglevel4", "algorithm", "feedback", "oscillatorkeysync", "lfospeed", "lfodelay", "lfopitchmodulationdepth", "lfoamplitudemodulationdepth", "lfokeysync", "lfowave", "lfopitchmodulationsensitivity", "transpose"]
    p_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    p_max = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 3, 3, 7, 3, 7, 99, 1, 31, 99, 14, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 3, 3, 7, 3, 7, 99, 1, 31, 99, 14, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 3, 3, 7, 3, 7, 99, 1, 31, 99, 14, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 3, 3, 7, 3, 7, 99, 1, 31, 99, 14, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 3, 3, 7, 3, 7, 99, 1, 31, 99, 14, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 3, 3, 7, 3, 7, 99, 1, 31, 99, 14, 99, 99, 99, 99, 99, 99, 99, 99, 31, 7, 1, 99, 99, 99, 99, 1, 5, 7, 48]

    params = [
        PatchParam(n, t, mn, mx)
        for n, t, mn, mx in zip(p_name, p_type, p_min, p_max)
    ]

    c_lengths = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 32, 6]

    c_idx = 0
    for p in params:
        if p.p_type in ('c', 'x'):
            p.n_classes = c_lengths[c_idx]
            c_idx += 1

    return params

def get_algorithms():
    algorithms = []
    with open("E:\\Coding\\vae-main\\dx7\\dx7.algorithms", "r") as f:
        for line in f:
            size = 7
            algorithm = [[1024 for _ in range(size)] for _ in range(size)]
            edges = line.split(",")
            for edge in edges:
                start = int(edge[0]) - 1
                end = int(edge[1]) - 1

                algorithm[start][end] = 1
            algorithms.append(algorithm)

    all_graphs = []
    for dist in algorithms:
        n = len(dist)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        all_graphs.append(dist)

    alibi_distances = -1 * (torch.tensor(all_graphs)) + 1
    algorithms = torch.tensor(algorithms)
    algorithms = algorithms.where(algorithms == 1, 0)
    return algorithms, alibi_distances

class PatchProcessor:
    def __init__(self, params):
        self.params = params

    def normalize(self, df):
        df = df.copy()

        for p in self.params:
            if p.p_type == 'm':
                df[p.name] = (
                    df[p.name].clip(lower=p.p_min) / p.p_max
                )

        return df

    def one_hot_dataframe(self, df):
        columns = []
        new_types = []

        for p in self.params:
            col = df[p.name]

            if p.p_type == 'm':
                columns.append(col)
                new_types.append('m')

            elif p.p_type == 'b':
                d = pd.get_dummies(col, prefix=p.name)
                columns.append(d)
                new_types.extend(['b'] * d.shape[1])

            elif p.p_type in ('c', 'x'):
                d = pd.get_dummies(col, prefix=p.name)

                expected = [f"{p.name}_{i}" for i in range(p.n_classes)]
                for e in expected:
                    if e not in d.columns:
                        d[e] = 0

                d = d[expected]
                columns.append(d)

                new_types.extend([p.p_type] * p.n_classes)

        df_out = pd.concat(columns, axis=1).astype('float32')
        return df_out, new_types

    def make_masks(self, expanded_types):
        t = torch.tensor([ord(c) for c in expanded_types], dtype=torch.int32)

        return (
            t == ord('m'),
            t == ord('b'),
            t == ord('c'),
            t == ord('x')
        )
