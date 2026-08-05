# Continuous Distillation Data Visualizer

A Streamlit application for browsing the time-series and concentration data
recorded on the continuous distillation mini-plant at the Laboratory for
Chemical Process Engineering, TUM Campus Straubing.

A hosted instance runs at
<https://continuousdistillationtum.streamlit.app/>.

## What it does

- Select a scenario (water, n-butanol/water, or OME), an operating point, and a
  run, and view its metadata alongside the data.
- Browse the raw table, filter it, and download the CSV.
- Plot any combination of sensors as a line plot, rolling average, heatmap,
  autocorrelation, or seasonal decomposition.
- Shade the periods marked by any of the anomaly label columns.
- For the n-butanol/water runs, plot measured concentration profiles against
  packing height.
- Look up a sensor or actuator tag in the scenario's feature description file.

## Data

The app does not bundle the data. Download the dataset from Zenodo
(<https://doi.org/10.5281/zenodo.17628963>) and unzip it, then either

- place the unzipped folder next to `app.py` under the name
  `ContinuousDistillationData`, or
- point the `DATA_ROOT` environment variable at it.

The expected layout is the one used in the deposit:

```
ContinuousDistillationData/
  ScenarioA_SingleComponent_Water/
    FeaturesOverview_Water.csv
    operating_point_001/
      train_normal_experiment_001.csv
      train_normal_experiment_001_metadata.yaml
      test_anormal_experiment_001.csv
      ...
  ScenarioB_BinaryComponent_n-butanolwater/
  ScenarioC_Reactive_OME/
```

Metadata files are picked up as either `<run>_metadata.yaml` or `<run>.yaml`.

The P&ID shown at the top of the page is read from `PID.png` in the working
directory. If the file is absent the app runs normally without it.

## Running locally

```bash
git clone https://github.com/aparnamuraleedharan/datavisualization.git
cd datavisualization
pip install -r requirements.txt
export DATA_ROOT=/path/to/ContinuousDistillationData   # optional
streamlit run app.py
```

Python 3.10 or later.

## Notes on the data

Times in the CSV files are clock times (HH:MM:SS) with no date, and rows within
a file run consecutively. The app detects runs that cross midnight and unwraps
them onto consecutive days so that they sort and plot correctly. The dates shown
are therefore artificial and carry no meaning beyond ordering.

The weighing-scale channels (`A...`, `M...`) contain occasional spikes that occur
during normal operation and are not labelled as anomalies. They should be
removed before computing mean flows or other statistics.

## Related

- Dataset: <https://doi.org/10.5281/zenodo.17628963>
- Preprocessing code:
  <https://github.com/aparnamuraleedharan/DataProcessing>

## License

MIT. See `LICENSE`.
