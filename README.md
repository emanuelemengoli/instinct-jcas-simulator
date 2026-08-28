# JCAS Simulator

A configurable, toroidal (periodic) joint communication-and-sensing (JCAS) network simulator: base stations, mobile UEs, and sensing objects on a Poisson or fixed-count network, with a Voronoi-tessellated coverage map, RT/Rayleigh-fading channel models, Kalman/Extended Kalman filtering, optional sector beamforming and TDD scheduling, and a separate simplified "non-captive" tracking model.

![JCAS network realization and Voronoi tessellation](images/readme_network_example.png)

You can drive it two ways:

- **The interactive web app** (`simulator_interface.py`) — configure a scenario with sliders and dropdowns, run it, and explore the results in your browser. No Jupyter, no cloning required once it's deployed.
- **The notebook** (`main.ipynb`) — the full configuration-driven Python API, for scripted experiments, channel-law ablations, and anything not exposed in the app's UI.

## Quick start

```bash
git clone <this-repo-url>
cd jcas_simulator
./setup.sh
source .venv/bin/activate
streamlit run simulator_interface.py
```

`setup.sh` creates a `.venv` in the repo root and installs everything in `requirements.txt` (numpy, scipy, shapely, matplotlib, seaborn, Jupyter, pytest, Streamlit). It's safe to re-run — it reuses an existing `.venv` instead of recreating it. On Windows, run the equivalent manually:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## The interactive app

```bash
streamlit run simulator_interface.py
```

Pick a scenario in the sidebar — the **captive scenario** (the full large-scale network simulator) or the **non-captive toy model** (a simplified single-track JCAS-vs-sensing-only tracking experiment) — configure its parameters, and click **Run simulation**. Results are organized into tabs: network/Voronoi view, SINR & filtering distributions, communication/sensing association scatter plots, an animated GIF of entity trajectories, and a summary.

Every run's Summary tab has an **Export** section:
- **Download all images (ZIP)** — every plot generated for that run (plus the trajectory animation, if you generated one), as PNGs.
- **Download parameters (LaTeX)** — the exact parameters used for that run, as a small standalone `.tex` file (a `booktabs` table) that compiles as-is with `pdflatex`.

The ray-traced (`rt`) channel model is significantly slower than the Rayleigh-fading (`exponential`) one — the app warns you before running it.

## The notebook

```bash
jupyter notebook main.ipynb
```

`main.ipynb` exercises the full `SimulationConfig` API directly: region/network/mobility/channel/filtering/beamforming/TDD configuration, the RT-vs-exponential channel-law ablation, and the non-captive tracking model, with every plotting function from `jcas_simulator.visualization` available for finer control than the app's UI exposes (custom `save_path`s, log-axis scatter plots, trajectory subsetting, etc.).

## Project layout

```
simulator_interface.py  Streamlit app
main.ipynb             Full config-API notebook
setup.sh                Creates .venv and installs dependencies
requirements.txt       Python dependencies
jcas_simulator/         The simulator package
  config.py              All configuration dataclasses (SimulationConfig, ...)
  simulator.py            Orchestration (JCASSimulator, LargeScaleJCASSimulator)
  network.py, geometry.py Network generation, toroidal Voronoi tessellation
  channel.py              Physical channel models (Rayleigh-fading, ray-traced)
  mobility.py, filtering.py   Motion models, Kalman/EKF filtering
  beamforming.py, scheduling.py  Optional sector beamforming, TDD scheduling
  non_captive/             The simplified non-captive tracking model
  visualization/           Plotting functions (KDEs, scatter plots, trajectories, animation)
tests/                  pytest suite
```

## Configuration

Every simulation is described by one `SimulationConfig` — a frozen dataclass tree (region, network, mobility, channel, filtering, beamforming, TDD, and non-captive-model settings). See `jcas_simulator/config.py` for every field and `main.ipynb` for worked examples; the app's sidebar exposes the most commonly changed subset of it.

## Testing

```bash
source .venv/bin/activate
pytest
```

## Deploying the app

The app has no external services or secrets, so it deploys as-is to [Streamlit Community Cloud](https://streamlit.io/cloud) for free: push this repo to GitHub, connect it on Streamlit Community Cloud, and point it at `simulator_interface.py`. Any other host that can run `pip install -r requirements.txt && streamlit run simulator_interface.py` works too.

## License

No license file is included yet. Add one (e.g. MIT, Apache-2.0) before treating this as open for reuse — until then, all rights are reserved by default.
