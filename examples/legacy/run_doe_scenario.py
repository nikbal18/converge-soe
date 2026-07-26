#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import logging

import pandas as pd

import converge_soe as csoe

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scaling', type=float, default='1.0')
parser.add_argument('indir', type=str)
parser.add_argument('outdir', type=str)

args = parser.parse_args()
scaling = args.scaling
indir = Path(args.indir)
outdir = Path(args.outdir)

outdir.mkdir(exist_ok=True, parents=True)

with (indir / 'network.json').open() as f:
    netw = json.load(f)

forecast = pd.read_csv(
    indir / 'forecast.csv', dtype={'load_id': str}
).set_index('load_id')
forecast *= scaling

csoe.logger.addHandler(logging.FileHandler(outdir / 'log.txt'))

solver = csoe.DoeSolver(netw, forecast, envelope_abs_max=50.0, solver_options={})
status, results = solver.solve()

#solver.dump_opt_model(outdir / 'model.txt')
results.soe.to_csv(outdir / 'doe.csv')
results.viol.to_csv(outdir / 'viol.csv')
results.bus.to_csv(outdir / 'bus.csv')
results.branch.to_csv(outdir / 'branch.csv')
