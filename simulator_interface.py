"""Interactive Streamlit front-end for the JCAS simulator.

Lets a visitor configure a scenario and see the same plots as main.ipynb in
a browser, without installing Jupyter or cloning the repo.
"""

from __future__ import annotations

import io
import math
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from jcas_simulator import (
    BeamformingConfig,
    ChannelConfig,
    CommunicationConfig,
    FilterConfig,
    JCASSimulator,
    MotionConfig,
    NetworkConfig,
    NonCaptiveConfig,
    ObservationConfig,
    PointProcessConfig,
    PopulationConfig,
    RegionConfig,
    SimulationConfig,
    TDDConfig,
    TDDPhaseConfig,
)
from jcas_simulator.visualization import (
    animate_entity_trajectories,
    plot_covariance_trace_kde,
    plot_filter_queue_association_scatter,
    plot_interference_association_scatter,
    plot_non_captive_estimation_comparison,
    plot_sinr_association_scatter,
    plot_sinr_kde,
    plot_voronoi_network,
    plot_workload_kde,
)

st.set_page_config(page_title="JCAS Simulator", layout="wide")


@st.cache_data(show_spinner=False)
def run_large_scale(
    seed: int,
    width: float,
    height: float,
    bs_count: int,
    bs_count_model: str,
    horizon: int,
    channel_model: str,
    placement: str,
    ue_per_cell: int,
    so_per_cell: int,
    population_count_model: str,
    placement_std_fraction: float,
    arrival_rate: float,
    motion_kind: str,
    rho: float,
    process_noise_std: float,
    state_dim: int,
    initial_speed_min_mps: float,
    initial_speed_max_mps: float,
    transmit_power_dbm: float,
    noise_psd_dbm_per_hz: float,
    bandwidth_hz: float,
    carrier_frequency_hz: float,
    exponential_path_loss_exponent: float,
    exponential_fading_mean: float,
    exponential_min_distance_m: float,
    radar_cross_section: float,
    filter_kind: str,
    observation_kind: str,
    observation_dim: int,
    initial_state_error_std: float,
    range_std: float,
    bearing_std_rad: float,
    range_rate_std: float,
    beamforming_enabled: bool,
    main_lobe_gain_db: float,
    side_lobe_gain_db: float,
    log2_beams: int,
    tdd_enabled: bool,
):
    motion = MotionConfig(
        kind=motion_kind,
        state_dim=state_dim,
        rho=rho,
        process_noise_std=process_noise_std,
        initial_speed_min_mps=initial_speed_min_mps,
        initial_speed_max_mps=initial_speed_max_mps,
    )
    if bs_count_model == "poisson":
        base_stations = PointProcessConfig(
            fixed_count=None, intensity_per_m2=bs_count / (width * height)
        )
    else:
        base_stations = PointProcessConfig(fixed_count=bs_count)

    if population_count_model == "poisson":
        ue_population = PopulationConfig(
            count_model="poisson",
            mean_per_cell=float(ue_per_cell),
            placement=placement,
            placement_std_fraction=placement_std_fraction,
        )
        so_population = PopulationConfig(
            count_model="poisson",
            mean_per_cell=float(so_per_cell),
            placement=placement,
            placement_std_fraction=placement_std_fraction,
        )
    else:
        ue_population = PopulationConfig(
            count_model="fixed",
            fixed_per_cell=ue_per_cell,
            placement=placement,
            placement_std_fraction=placement_std_fraction,
        )
        so_population = PopulationConfig(
            count_model="fixed",
            fixed_per_cell=so_per_cell,
            placement=placement,
            placement_std_fraction=placement_std_fraction,
        )

    config = SimulationConfig(
        master_seed=seed,
        horizon=horizon,
        network=NetworkConfig(
            region=RegionConfig(width=width, height=height),
            base_stations=base_stations,
            ue_population=ue_population,
            so_population=so_population,
        ),
        ue_motion=motion,
        so_motion=motion,
        channel=ChannelConfig(
            model=channel_model,
            transmit_power_dbm=transmit_power_dbm,
            noise_psd_dbm_per_hz=noise_psd_dbm_per_hz,
            bandwidth_hz=bandwidth_hz,
            carrier_frequency_hz=carrier_frequency_hz,
            exponential_path_loss_exponent=exponential_path_loss_exponent,
            exponential_fading_mean=exponential_fading_mean,
            exponential_min_distance_m=exponential_min_distance_m,
            radar_cross_section=radar_cross_section,
        ),
        communication=CommunicationConfig(arrival_rate=arrival_rate),
        filtering=FilterConfig(
            kind=filter_kind,
            initial_state_error_std=initial_state_error_std,
            observation=ObservationConfig(
                kind=observation_kind,
                observation_dim=observation_dim,
                range_std=range_std,
                bearing_std_rad=bearing_std_rad,
                range_rate_std=range_rate_std,
            ),
        ),
        beamforming=BeamformingConfig(
            enabled=beamforming_enabled,
            main_lobe_gain_db=main_lobe_gain_db,
            side_lobe_gain_db=side_lobe_gain_db,
            log2_beams=log2_beams,
        ),
        tdd=TDDConfig(
            enabled=tdd_enabled,
            phases=(
                TDDPhaseConfig("communication", 2, True, False),
                TDDPhaseConfig("sensing", 1, False, True),
            ),
        )
        if tdd_enabled
        else TDDConfig(),
    )
    return JCASSimulator(config).run()


@st.cache_data(show_spinner=False)
def run_non_captive(
    seed: int,
    horizon: int,
    smoothing_window: int,
    delta: float,
    position_process_variance: float,
    gain_process_variance: float,
    path_loss_exponent: float,
    gain_rho: float,
):
    config = SimulationConfig(
        master_seed=seed,
        operation_mode="non_captive",
        non_captive=NonCaptiveConfig(
            horizon=horizon,
            smoothing_window=smoothing_window,
            delta=delta,
            position_process_variance=position_process_variance,
            gain_process_variance=gain_process_variance,
            path_loss_exponent=path_loss_exponent,
            gain_rho=gain_rho,
        ),
    )
    return JCASSimulator(config).run()


def show(fig, name: str, collector: list[tuple[str, bytes]] | None = None) -> None:
    """Render a figure and, if a collector list is given, capture it as PNG bytes."""
    st.pyplot(fig, clear_figure=False)
    if collector is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        collector.append((name, buf.getvalue()))
    plt.close(fig)


def _latex_escape(value: object) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def params_to_latex(mode: str, params: dict[str, object]) -> str:
    """Render a run's parameters as a small standalone, compilable LaTeX document."""
    scenario = "Captive scenario" if mode == "captive" else "Non-captive toy model"
    rows = "\n".join(
        f"{_latex_escape(label)} & {_latex_escape(value)} \\\\" for label, value in params.items()
    )
    return (
        "\\documentclass{article}\n"
        "\\usepackage{booktabs}\n"
        "\\begin{document}\n\n"
        "\\begin{table}[h]\n"
        "\\centering\n"
        f"\\caption{{JCAS simulator parameters --- {_latex_escape(scenario)}}}\n"
        "\\begin{tabular}{ll}\n"
        "\\toprule\n"
        "Parameter & Value \\\\\n"
        "\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
        "\\end{document}\n"
    )


INTERPRETATION_NOTES = r"""
#### Association ratio

Let $X,Y\geq 0$ denote two jointly observed steady-state (or full trajectories comprising transient dynamics) performance metrics, for instance communication and sensing interference, communication and sensing SINR, or queue workload and estimation uncertainty. From paired observations $\{(X_n,Y_n)\}_{n=1}^N$, the reported statistic is

$$
\widehat{\mathcal A}_{X,Y}
=
\frac{N^{-1}\sum_{n=1}^N X_nY_n}
{\left(N^{-1}\sum_{n=1}^N X_n\right)
 \left(N^{-1}\sum_{n=1}^N Y_n\right)} .
$$

Its population counterpart is

$$
\mathcal A_{X,Y}
=
\frac{\mathbb E[XY]}{\mathbb E[X]\mathbb E[Y]},
\qquad
0<\mathbb E[X]\mathbb E[Y]<\infty,
\quad
\mathbb E[XY]<\infty .
$$

Equivalently,

$$
\mathcal A_{X,Y}-1
=
\frac{\operatorname{Cov}(X,Y)}
{\mathbb E[X]\mathbb E[Y]} .
$$

Hence $\mathcal A_{X,Y}>1$ corresponds to positive covariance normalized by the product of the marginal means, while $\mathcal A_{X,Y}<1$ corresponds to negative covariance. Under independence, $\mathcal A_{X,Y}=1$, but the converse is false.

The statistic is dimensionless and invariant under positive multiplicative rescaling,

$$
\mathcal A_{aX,bY}
=
\mathcal A_{X,Y},
\qquad a,b>0,
$$

but not under translations. It should therefore be interpreted as a normalized first-order co-moment, not as a distributional distance.

The association ratio is only diagnostic. It does not prove positive association in the stronger sense

$$
\operatorname{Cov}(f(X),g(Y))\geq 0
$$

for all coordinatewise non-decreasing measurable functions $f$ and $g$. The inequality $\mathcal A_{X,Y}>1$ verifies this property only for the particular choice $f(x)=x$ and $g(y)=y$.

#### Relation with Pearson correlation

Pearson correlation and the association ratio use the same covariance but different normalizations:

$$
\rho_{X,Y}
=
\frac{\operatorname{Cov}(X,Y)}
{\sqrt{\operatorname{Var}(X)\operatorname{Var}(Y)}},
\qquad
\mathcal A_{X,Y}-1
=
\frac{\operatorname{Cov}(X,Y)}
{\mathbb E[X]\mathbb E[Y]} .
$$

Pearson correlation requires finite, non-zero marginal variances. The association ratio requires finite first marginal moments and a finite mixed moment. It may therefore remain well defined in some infinite-variance regimes, provided $\mathbb E[XY]<\infty$.

This should not be interpreted as robustness to heavy tails. If the mixed moment is infinite, or if the empirical mixed moment is dominated by rare extreme observations, $\widehat{\mathcal A}_{X,Y}$ may be unstable. Moreover, both $\rho_{X,Y}=0$ and $\mathcal A_{X,Y}=1$ are compatible with nonlinear statistical dependence.

#### Scatter-plot representation

The scatter plot displays the empirical support of the paired observations $(X_n,Y_n)$. It should be used to assess the geometric structure underlying $\widehat{\mathcal A}_{X,Y}$: monotone trends, clustering, saturation, heteroscedasticity, and isolated extreme points may lead to different interpretations of the same scalar value.

Axis scaling is purely representational. It does not modify the observations and does not enter the computation of $\widehat{\mathcal A}_{X,Y}$. When one coordinate spans several orders of magnitude, the corresponding axis may be displayed logarithmically. Since the two coordinates may have different empirical ranges, the choice of scale is axis-specific:

$$
(x,y),\qquad
(\log x,y),\qquad
(x,\log y),\qquad
(\log x,\log y)
$$

are admissible visual representations, selected independently according to the spread of each marginal sample.

A logarithmic axis is appropriate when a small number of large values would otherwise compress most observations into an unreadable region of the plot. It should not be interpreted as changing the dependence measure; it changes only the geometry by which the empirical cloud is inspected.

#### Kernel density estimates

For scalar observations $X_1,\ldots,X_N$, the KDE

$$
\widehat f_h(x)
=
\frac{1}{Nh}
\sum_{n=1}^N
K\!\left(\frac{x-X_n}{h}\right)
$$

estimates the marginal density of the corresponding simulated quantity. It is used only to summarize one-dimensional marginal behavior: location, dispersion, modality, and tail weight. It is not a joint-dependence diagnostic.

Regions where $\widehat f_h(x)$ is large correspond to values that occur frequently under the simulated steady-state regime. Local maxima identify modes of the empirical distribution, while the spread of the density describes the variability of the quantity. Long or slowly decaying tails indicate the occurrence of comparatively rare but potentially large values.

The density value is not itself a probability; probabilities correspond to integrated mass,

$$
\mathbb P(X\in B)
\approx
\int_B \widehat f_h(x)\,dx .
$$

The bandwidth $h$ controls the degree of smoothing. A small bandwidth produces a more variable estimate and may display sampling fluctuations as apparent modes, whereas a large bandwidth may suppress genuine distributional structure.

Consequently, KDEs should be interpreted qualitatively unless bandwidth choice and sample size are explicitly controlled.

#### Interpretation in the simulator

Scatter plots and association ratios describe joint behavior; KDEs describe marginal behavior. They therefore answer distinct questions.

For a pair $(X,Y)$, the condition

$$
\widehat{\mathcal A}_{X,Y}>1
$$

means that large values of $X$ and $Y$, on average, tend to occur together more strongly than predicted by the product of their empirical means. Values close to one indicate weak covariance at this normalization scale, not independence.

In the reported outputs,

$$
\widehat{\mathcal A}_{I_{\mathrm{sen}},I_{\mathrm{com}}}>1
$$

indicates positive co-fluctuation of sensing and communication interference,

$$
\widehat{\mathcal A}_{\mathrm{SINR}_{\mathrm{sen}},\mathrm{SINR}_{\mathrm{com}}}>1
$$

indicates that favorable or unfavorable propagation/network configurations affect both SINR metrics jointly, and

$$
\widehat{\mathcal A}_{W,\operatorname{Tr}(\Sigma)}>1
$$

indicates that communication congestion and estimation uncertainty co-increase in the simulated steady state.

All reported quantities are finite-sample estimates, spatially distributed, from temporally dependent trajectories. They should be interpreted as empirical approximations of steady-state objects, or of systems comprising transient dynamics (e.g. where unstable and stable queues coexist).
"""


st.title("JCAS Simulator")
st.caption(
    "Joint communication-and-sensing network simulator — run scenarios and "
    "explore the results in your browser, no notebook download required."
)

with st.sidebar:
    st.header("Scenario")
    operation_mode = st.radio(
        "Simulation strategy",
        ["captive", "non_captive"],
        format_func=lambda m: "Captive scenario"
        if m == "captive"
        else "Non-captive toy model",
    )
    seed = st.number_input(
        "Random seed",
        min_value=0,
        value=42,
        step=1,
        help="Controls the randomness of this run. The same seed always reproduces the exact same result.",
    )

    if operation_mode == "captive":
        st.subheader("Network")
        width = st.slider("Region width (m)", 50.0, 5000.0, 1500.0, step=50.0)
        height = st.slider("Region height (m)", 50.0, 5000.0, 1500.0, step=50.0)
        bs_count_model = st.radio(
            "Base station count",
            ["fixed", "poisson"],
            format_func=lambda m: "Fixed" if m == "fixed" else "Random (Poisson)",
            horizontal=True,
            help="A random count varies the number of base stations from run to run "
            "(drawn from a Poisson distribution) — useful for modeling real, unpredictable deployments.",
        )
        bs_count = st.slider(
            "Base stations" if bs_count_model == "fixed" else "Average base stations",
            5,
            300,
            15,
        )
        horizon = st.slider("Simulation length (time steps)", 10, 1000, 80, step=10)

        st.subheader("Channel")
        channel_model = st.selectbox(
            "Channel model",
            ["exponential", "rt"],
            format_func=lambda m: "Rayleigh fading (fast)"
            if m == "exponential"
            else "Ray-traced (slow, realistic)",
            help="How radio signals fade due to obstacles, reflections, and distance. Rayleigh "
            "fading is a fast statistical approximation; ray tracing simulates the physical "
            "propagation paths and is more realistic but much slower.",
        )
        if channel_model == "rt":
            st.caption(
                "Ray-traced channel is much slower to simulate — a 200-step, "
                "30-BS run can take roughly a minute."
            )

        # Defaults (ChannelConfig field defaults) for whichever model-specific
        # group isn't rendered below, so every variable is always defined.
        exponential_path_loss_exponent = 2.7
        exponential_fading_mean = 1.0
        exponential_min_distance_m = 1.0
        radar_cross_section = 1.0
        with st.expander("Channel: Advanced parameters"):
            transmit_power_dbm = st.slider(
                "Transmit power (dBm)",
                0.0,
                60.0,
                46.0,
                step=1.0,
                help="Signal strength leaving a base station, in dBm (a standard radio power unit). Higher is stronger.",
            )
            noise_psd_dbm_per_hz = st.slider(
                "Background noise level (dBm/Hz)",
                -200.0,
                -80.0,
                -174.0,
                step=1.0,
                help="How much random background (thermal) noise competes with the signal. "
                "Lower (more negative) means a quieter, cleaner environment.",
            )
            bandwidth_mhz = st.slider(
                "Bandwidth (MHz)",
                1.0,
                100.0,
                20.0,
                step=1.0,
                help="How wide a slice of radio spectrum is used. More bandwidth generally means a stronger signal-to-noise ratio.",
            )
            carrier_ghz = st.slider(
                "Carrier frequency (GHz)",
                0.1,
                10.0,
                1.0,
                step=0.1,
                help="The radio frequency the signal is transmitted on.",
            )
            bandwidth_hz = bandwidth_mhz * 1.0e6
            carrier_frequency_hz = carrier_ghz * 1.0e9
            if channel_model == "exponential":
                exponential_path_loss_exponent = st.slider(
                    "Path-loss exponent",
                    1.0,
                    5.0,
                    2.7,
                    step=0.1,
                    help="How quickly signal strength drops off as distance increases. Higher "
                    "means faster drop-off (2 = free space, 3-4 = typical dense/urban area).",
                )
                exponential_fading_mean = st.slider(
                    "Average fading strength",
                    0.1,
                    5.0,
                    1.0,
                    step=0.1,
                    help="The average strength of the random (Rayleigh) fading multiplier applied to the signal.",
                )
                exponential_min_distance_m = st.slider(
                    "Minimum sensor distance (m)",
                    0.1,
                    20.0,
                    1.0,
                    step=0.1,
                    help="A floor on the distance used in the signal-decay calculation, avoiding "
                    "an unrealistic power spike when two entities are very close together.",
                )
            else:
                radar_cross_section = st.slider(
                    "Radar cross-section (m²)",
                    0.1,
                    50.0,
                    1.0,
                    step=0.1,
                    help="How large/reflective the sensed object appears to radar — bigger, more "
                    "reflective objects (e.g. vehicles) have a larger cross-section than small ones (e.g. a person).",
                )

        st.subheader("Population & placement")
        placement = st.selectbox(
            "User & sensor placement",
            ["uniform", "gaussian_around_bs"],
            format_func=lambda m: "Uniformly distributed" if m == "uniform" else "Clustered near base stations",
            help="UE = user equipment (phones/devices); SO = sensing object (the target being tracked).",
        )
        placement_std_fraction = 0.20  # PopulationConfig default; unused unless clustered.
        if placement == "gaussian_around_bs":
            with st.expander("Population & placement: Advanced parameters"):
                placement_std_fraction = st.slider(
                    "Clustering tightness",
                    0.01,
                    1.0,
                    0.20,
                    step=0.01,
                    help="How tightly users/sensors cluster around their base station, as a "
                    "fraction of that base station's coverage-area size. Smaller values pack "
                    "them close to the base station; larger values spread them across most of the area.",
                )
        population_count_model = st.radio(
            "Population per cell",
            ["fixed", "poisson"],
            format_func=lambda m: "Fixed" if m == "fixed" else "Random (Poisson)",
            horizontal=True,
            help="A random count varies how many users/sensing objects each base station gets, "
            "drawn from a Poisson distribution, instead of exactly the same number every time.",
        )
        ue_label = "Users per cell" if population_count_model == "fixed" else "Average users per cell"
        so_label = (
            "Sensing objects per cell"
            if population_count_model == "fixed"
            else "Average sensing objects per cell"
        )
        ue_per_cell = st.slider(ue_label, 1, 100, 1)
        so_per_cell = st.slider(so_label, 1, 100, 1)

        st.subheader("Communication")
        with st.expander("Communication: Advanced parameters"):
            arrival_rate = st.slider(
                "Data traffic intensity",
                0.0,
                20.0,
                1.0,
                step=0.1,
                help="How often new communication requests arrive at each base station's queue. Higher means a busier network.",
            )

        st.subheader("Mobility")
        motion_kind = st.selectbox(
            "Movement pattern",
            ["static", "gauss_markov", "rho_random_walk"],
            index=1,
            format_func=lambda m: {
                "static": "Stationary (no movement)",
                "gauss_markov": "Gauss-Markov process",
                "rho_random_walk": "ρ-persistent random walk",
            }[m],
        )
        rho = st.slider(
            "Movement persistence",
            0.0,
            1.0,
            0.9,
            step=0.01,
            help="How much each step's movement continues the previous step's. 0 = completely "
            "random every step; close to 1 = smooth, realistic movement that keeps going the same way.",
        )
        process_noise_std = st.slider(
            "Movement randomness",
            0.0,
            5.0,
            0.2,
            step=0.05,
            help="The amount of random \u2018jitter\u2019 added to movement at each step.",
        )

        with st.expander("Mobility: Advanced parameters"):
            state_dim_forced = motion_kind == "rho_random_walk"
            state_dim = st.selectbox(
                "What's tracked",
                [2, 4],
                index=1 if state_dim_forced else 0,
                format_func=lambda d: "Position only (x, y)" if d == 2 else "Position + velocity (x, y, speed)",
                disabled=state_dim_forced,
                help="\"ρ-persistent random walk\" requires tracking velocity too."
                if state_dim_forced
                else None,
            )
            if state_dim_forced:
                state_dim = 4
            if state_dim == 4:
                initial_speed_min_mps, initial_speed_max_mps = st.slider(
                    "Initial speed range (m/s)", 0.0, 20.0, (0.0, 0.0), step=0.5
                )
                if motion_kind == "gauss_markov":
                    st.info(
                        "With the Gauss-Markov model the initial speed is only drawn "
                        "into the tracked state, so it affects parameter estimation "
                        "and not the real motion. Speed selection affects motion only "
                        "in the ρ-persistent random walk."
                    )
            else:
                initial_speed_min_mps, initial_speed_max_mps = 0.0, 0.0

        st.subheader("Filtering")
        filter_kind = st.selectbox(
            "Tracking filter",
            ["auto", "kf", "ekf"],
            format_func=lambda k: {
                "auto": "Automatic (recommended)",
                "kf": "Kalman filter (for direct-position measurements)",
                "ekf": "Extended Kalman filter (for radar-style measurements)",
            }[k],
        )
        observation_kind = st.selectbox(
            "What the sensor measures",
            ["linear", "range_bearing", "range_bearing_rate"],
            index=1,
            format_func=lambda k: {
                "linear": "Position directly",
                "range_bearing": "Distance & angle (radar-style)",
                "range_bearing_rate": "Distance, angle & closing speed (radar-style)",
            }[k],
        )

        # Defaults (ObservationConfig/FilterConfig field defaults) for
        # whichever conditional group below isn't rendered this run.
        observation_dim = 2
        range_std = 0.5
        bearing_std_rad = math.radians(1.0)
        range_rate_std = 0.1
        with st.expander("Filtering: Advanced parameters"):
            initial_state_error_std = st.slider(
                "Starting estimate error",
                0.0,
                10.0,
                0.0,
                step=0.5,
                help="How far off the tracker's very first guess is from the true position.",
            )
            if observation_kind == "linear":
                obs_dim_options = [2, 3] if state_dim == 4 else [2]
                observation_dim = st.selectbox("Number of measured coordinates", obs_dim_options)
            else:
                range_std = st.slider(
                    "Distance measurement noise",
                    0.0,
                    5.0,
                    0.5,
                    step=0.1,
                    help="Uncertainty (noise) in each distance measurement.",
                )
                bearing_std_deg = st.slider(
                    "Angle measurement noise (degrees)",
                    0.1,
                    10.0,
                    1.0,
                    step=0.1,
                    help="Uncertainty (noise) in each angle measurement.",
                )
                bearing_std_rad = math.radians(bearing_std_deg)
                if observation_kind == "range_bearing_rate":
                    range_rate_std = st.slider(
                        "Speed measurement noise",
                        0.0,
                        5.0,
                        0.1,
                        step=0.05,
                        help="Uncertainty (noise) in each closing-speed measurement.",
                    )

        st.subheader("Beamforming & scheduling")
        beamforming_enabled = st.checkbox(
            "Sector beamforming",
            value=False,
            help="Focuses a base station's signal toward specific users/devices instead of "
            "broadcasting equally in every direction, boosting signal strength in those directions.",
        )
        main_lobe_gain_db = 23.0
        side_lobe_gain_db = 3.0
        log2_beams = 4
        if beamforming_enabled:
            if channel_model == "rt":
                st.warning(
                    "Beamforming is only compatible with the exponential channel; "
                    "switching the channel model to exponential for this run."
                )
            with st.expander("Beamforming: Advanced parameters"):
                main_lobe_gain_db = st.slider(
                    "Signal boost toward target (dB)",
                    0.0,
                    40.0,
                    23.0,
                    step=1.0,
                )
                side_lobe_gain_db = st.slider(
                    "Signal leakage elsewhere (dB)",
                    0.0,
                    20.0,
                    3.0,
                    step=1.0,
                    help="How much signal still reaches directions outside the focused beam.",
                )
                beam_count = st.select_slider(
                    "Number of beams",
                    options=[1, 2, 4, 8, 16, 32, 64],
                    value=16,
                    help="How finely a base station can steer its focused beam — more beams means finer aim.",
                )
                log2_beams = int(math.log2(beam_count))
        tdd_enabled = st.checkbox(
            "TDD - communication & sensing",
            value=False,
            help="Instead of doing both at once every time step, split time into a repeating "
            "communication/sensing cycle (Time Division Duplex (TDD)).",
        )
    else:
        st.subheader("Non-captive toy model")
        nc_horizon = st.slider("Simulation length (time steps)", 50, 2000, 200, step=50)
        nc_smoothing = st.slider(
            "Smoothing window",
            5,
            200,
            20,
            step=5,
            help="How many recent time steps are averaged together when smoothing the error curves shown in the plots.",
        )
        nc_delta = st.slider(
            "Time step size",
            0.1,
            5.0,
            1.0,
            step=0.1,
            help="How much simulated time passes at each step — larger values mean the target "
            "moves further, and passes base stations faster, between steps.",
        )
        with st.expander("Non-captive: Advanced parameters"):
            nc_position_process_variance = st.slider(
                "Position randomness", 0.0, 200.0, 50.0, step=5.0
            )
            nc_gain_process_variance = st.slider(
                "Signal strength randomness", 0.0, 200.0, 50.0, step=5.0
            )
            nc_path_loss_exponent = st.slider(
                "Path-loss exponent", 0.5, 3.0, 1.2, step=0.1
            )
            nc_gain_rho = st.slider(
                "Signal strength persistence",
                0.0,
                1.0,
                0.9,
                step=0.01,
                help="How much each step's signal strength continues the previous step's. "
                "0 = completely random every step; close to 1 = smooth, slowly-changing signal strength.",
            )

    run_clicked = st.button("Run simulation", type="primary", use_container_width=True)

if run_clicked:
    try:
        if operation_mode == "captive":
            effective_channel = (
                "exponential" if beamforming_enabled and channel_model == "rt" else channel_model
            )
            spinner_msg = (
                "Running ray-traced simulation — this can take up to a minute..."
                if effective_channel == "rt"
                else "Running simulation..."
            )
            with st.spinner(spinner_msg):
                result = run_large_scale(
                    seed=seed,
                    width=width,
                    height=height,
                    bs_count=bs_count,
                    bs_count_model=bs_count_model,
                    horizon=horizon,
                    channel_model=effective_channel,
                    placement=placement,
                    ue_per_cell=ue_per_cell,
                    so_per_cell=so_per_cell,
                    population_count_model=population_count_model,
                    placement_std_fraction=placement_std_fraction,
                    arrival_rate=arrival_rate,
                    motion_kind=motion_kind,
                    rho=rho,
                    process_noise_std=process_noise_std,
                    state_dim=state_dim,
                    initial_speed_min_mps=initial_speed_min_mps,
                    initial_speed_max_mps=initial_speed_max_mps,
                    transmit_power_dbm=transmit_power_dbm,
                    noise_psd_dbm_per_hz=noise_psd_dbm_per_hz,
                    bandwidth_hz=bandwidth_hz,
                    carrier_frequency_hz=carrier_frequency_hz,
                    exponential_path_loss_exponent=exponential_path_loss_exponent,
                    exponential_fading_mean=exponential_fading_mean,
                    exponential_min_distance_m=exponential_min_distance_m,
                    radar_cross_section=radar_cross_section,
                    filter_kind=filter_kind,
                    observation_kind=observation_kind,
                    observation_dim=observation_dim,
                    initial_state_error_std=initial_state_error_std,
                    range_std=range_std,
                    bearing_std_rad=bearing_std_rad,
                    range_rate_std=range_rate_std,
                    beamforming_enabled=beamforming_enabled,
                    main_lobe_gain_db=main_lobe_gain_db,
                    side_lobe_gain_db=side_lobe_gain_db,
                    log2_beams=log2_beams,
                    tdd_enabled=tdd_enabled,
                )
            run_params = {
                "Random seed": seed,
                "Region width (m)": width,
                "Region height (m)": height,
                "Base station count model": bs_count_model,
                (
                    "Base stations" if bs_count_model == "fixed" else "Average base stations"
                ): bs_count,
                "Simulation length (time steps)": horizon,
                "Channel model": effective_channel,
                "Transmit power (dBm)": transmit_power_dbm,
                "Background noise level (dBm/Hz)": noise_psd_dbm_per_hz,
                "Bandwidth (Hz)": bandwidth_hz,
                "Carrier frequency (Hz)": carrier_frequency_hz,
                "Path-loss exponent": exponential_path_loss_exponent,
                "Average fading strength": exponential_fading_mean,
                "Minimum sensor distance (m)": exponential_min_distance_m,
                "Radar cross-section (m^2)": radar_cross_section,
                "User & sensor placement": placement,
                "Clustering tightness": placement_std_fraction,
                "Population count model": population_count_model,
                (
                    "Users per cell" if population_count_model == "fixed" else "Average users per cell"
                ): ue_per_cell,
                (
                    "Sensing objects per cell"
                    if population_count_model == "fixed"
                    else "Average sensing objects per cell"
                ): so_per_cell,
                "Data traffic intensity": arrival_rate,
                "Movement pattern": motion_kind,
                "Movement persistence": rho,
                "Movement randomness": process_noise_std,
                "Tracked state dimension": state_dim,
                "Initial speed range (m/s)": f"{initial_speed_min_mps}-{initial_speed_max_mps}",
                "Tracking filter": filter_kind,
                "What the sensor measures": observation_kind,
                "Number of measured coordinates": observation_dim,
                "Starting estimate error": initial_state_error_std,
                "Distance measurement noise": range_std,
                "Angle measurement noise (rad)": bearing_std_rad,
                "Speed measurement noise": range_rate_std,
                "Sector beamforming": beamforming_enabled,
                "Signal boost toward target (dB)": main_lobe_gain_db,
                "Signal leakage elsewhere (dB)": side_lobe_gain_db,
                "Number of beams": 2 ** log2_beams,
                "Alternate communication & sensing (TDD)": tdd_enabled,
            }
        else:
            with st.spinner("Running non-captive toy model simulation..."):
                result = run_non_captive(
                    seed,
                    nc_horizon,
                    nc_smoothing,
                    nc_delta,
                    nc_position_process_variance,
                    nc_gain_process_variance,
                    nc_path_loss_exponent,
                    nc_gain_rho,
                )
            run_params = {
                "Random seed": seed,
                "Simulation length (time steps)": nc_horizon,
                "Smoothing window": nc_smoothing,
                "Time step size": nc_delta,
                "Position randomness": nc_position_process_variance,
                "Signal strength randomness": nc_gain_process_variance,
                "Path-loss exponent": nc_path_loss_exponent,
                "Signal strength persistence": nc_gain_rho,
            }
        st.session_state["result"] = result
        st.session_state["result_mode"] = operation_mode
        st.session_state["run_params"] = run_params
    except ValueError as exc:
        st.error(f"Invalid configuration: {exc}")

result = st.session_state.get("result")
result_mode = st.session_state.get("result_mode")

if result is None:
    st.info("Configure a scenario in the sidebar, then click **Run simulation**.")
elif result_mode == "captive":
    collected_images: list[tuple[str, bytes]] = []
    tabs = st.tabs(["Network", "SINR & filtering", "Association", "Trajectory animation", "Summary"])

    with tabs[0]:
        max_index = len(next(iter(result.ue_trajectories.values()))) - 1 if result.ue_trajectories else 0
        time_index = st.slider("Time step to display", 0, max(max_index, 0), 0)
        fig, _ = plot_voronoi_network(result, time_index=time_index, show=False)
        show(fig, "network_voronoi", collected_images)

    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            fig, _ = plot_sinr_kde(result, steady_state='auto', show=False)
            show(fig, "sinr_kde", collected_images)
            fig, _ = plot_workload_kde(result, steady_state='auto', show=False)
            show(fig, "workload_kde", collected_images)
        with col2:
            fig, _ = plot_covariance_trace_kde(result, steady_state='auto', show=False)
            show(fig, "covariance_trace_kde", collected_images)

    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            fig, _ = plot_interference_association_scatter(
                result, show=False, xscale="log", yscale="log"
            )
            show(fig, "interference_association", collected_images)
            fig, _ = plot_filter_queue_association_scatter(result, show=False)
            show(fig, "filter_queue_association", collected_images)
        with col2:
            fig, _ = plot_sinr_association_scatter(result, show=False, xscale="log", yscale="log")
            show(fig, "sinr_association", collected_images)

    with tabs[3]:
        st.caption("Entity positions animated over time, with a short fading trail.")
        cache_key = id(result)
        if st.session_state.get("trajectory_gif_for") != cache_key:
            st.session_state.pop("trajectory_gif", None)

        if st.session_state.get("trajectory_gif") is None:
            if st.button("Generate trajectory animation (GIF)"):
                with st.spinner("Rendering trajectory animation..."):
                    st.session_state["trajectory_gif"] = animate_entity_trajectories(result)
                    st.session_state["trajectory_gif_for"] = cache_key
                st.rerun()
            else:
                st.info("Rendering takes a few seconds and isn't run automatically.")
        else:
            st.image(st.session_state["trajectory_gif"])
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "Download animation (GIF)",
                    data=st.session_state["trajectory_gif"],
                    file_name="trajectory_animation.gif",
                    mime="image/gif",
                    key="download_trajectory_gif",
                )
            with col_b:
                if st.button("Regenerate"):
                    st.session_state.pop("trajectory_gif", None)
                    st.rerun()

    with tabs[4]:
        summary = result.summary()
        col1, col2, col3 = st.columns(3)
        col1.metric("Base stations", summary["n_base_stations"])
        col2.metric("UEs", summary["n_ues"])
        col3.metric("Sensing objects", summary["n_sensing_objects"])
        st.write(
            f"Beamforming: {'enabled' if summary['beamforming_enabled'] else 'disabled'} · "
            f"TDD: {'enabled' if summary['tdd_enabled'] else 'disabled'}"
        )
        with st.expander("📖 Interpretation of simulation outputs"):
            st.markdown(INTERPRETATION_NOTES)

        st.subheader("Association metrics")
        st.json(summary["association"])

    st.divider()
    st.subheader("Export")
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        gif_bytes = st.session_state.get("trajectory_gif")
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in collected_images:
                zf.writestr(f"{name}.png", data)
            if gif_bytes is not None:
                zf.writestr("trajectory_animation.gif", gif_bytes)
        st.download_button(
            "Download all images (ZIP)",
            data=zip_buf.getvalue(),
            file_name="jcas_simulation_images.zip",
            mime="application/zip",
            key="download_images_zip",
        )
    with export_col2:
        latex_content = params_to_latex(result_mode, st.session_state.get("run_params", {}))
        st.download_button(
            "Download parameters (LaTeX)",
            data=latex_content.encode("utf-8"),
            file_name="jcas_parameters.tex",
            mime="text/x-tex",
            key="download_params_tex",
        )

else:  # non_captive
    collected_images = []
    tabs = st.tabs(["Estimation comparison", "SINR & covariance", "Summary"])

    with tabs[0]:
        fig, _ = plot_non_captive_estimation_comparison(result, show=False)
        show(fig, "non_captive_estimation_comparison", collected_images)

    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            fig, _ = plot_sinr_kde(result, steady_state='auto', show=False)
            show(fig, "non_captive_sinr_kde", collected_images)
        with col2:
            fig, _ = plot_covariance_trace_kde(result, steady_state='auto', show=False)
            show(fig, "non_captive_covariance_trace_kde", collected_images)

    with tabs[2]:
        n_steps = int(result.true_state.shape[1])
        col1, col2, col3 = st.columns(3)
        col1.metric("Time steps simulated", n_steps)
        col2.metric("Final JCAS position error", f"{float(result.smoothed_position_error_jcas[-1]):.3f}")
        col3.metric(
            "Final sensing-only position error",
            f"{float(result.smoothed_position_error_sensing_only[-1]):.3f}",
        )
        st.metric(
            "JCAS error vs. sensing-only baseline",
            f"{float(result.percent_error[-1]):.1f}%",
            help="Negative means JCAS tracking has lower position error than the sensing-only baseline.",
        )

    st.divider()
    st.subheader("Export")
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in collected_images:
                zf.writestr(f"{name}.png", data)
        st.download_button(
            "Download all images (ZIP)",
            data=zip_buf.getvalue(),
            file_name="jcas_simulation_images.zip",
            mime="application/zip",
            key="download_images_zip",
        )
    with export_col2:
        latex_content = params_to_latex(result_mode, st.session_state.get("run_params", {}))
        st.download_button(
            "Download parameters (LaTeX)",
            data=latex_content.encode("utf-8"),
            file_name="jcas_parameters.tex",
            mime="text/x-tex",
            key="download_params_tex",
        )

st.sidebar.divider()
st.sidebar.caption(
    "For the full config API and channel-law ablations, see main.ipynb in this repo."
)
