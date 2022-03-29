# reefline

Code and data in support of the Reefline experiment.

## Getting started

### Get the code

```
git clone https://github.com/sustain-lab/reefline
cd reefline
```

### Set up a Python environment

```
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -U -r requirements
```

### Run the code

Baseline experiments (no reef model):

```
python3 process_reefline_timeseries_baseline.py
```

Experiments with the model:

```
python3 process_reefline_timeseries.py
```

Both scripts will produce and PNG plot and a CSV processed elevation dataset
for each condition.
