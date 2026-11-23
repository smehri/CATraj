"""Interactive trajectory annotation.

A Panel app over :mod:`catraj.annotation`: load trajectories and context
observations, compare interpolation methods by cross-validation, annotate, and
export the result.

The method applies to any moving object -- vessels, aircraft, vehicles,
wildlife, pedestrians -- and to whatever context was measured around it. The
maritime sample data is one application, not the scope.

Run it with::

    panel serve 04_app/annotation_app.py --show

The workflow mirrors the method, one tab per stage:

**Overview**   the method as a diagram; click a stage to jump to its tab
**Data**       load a trajectory file and a station file, map the columns
**Spatial**    compare methods by leave-one-station-out, inspect surfaces
**Temporal**   compare methods by k-fold
**Annotate**   apply the chosen methods and export

Nothing is inferred that the method leaves to the user: which variables are
bearings, and which interpolators to use, are choices the app surfaces rather
than guesses.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import geopandas as gpd
import holoviews as hv
import hvplot.pandas  # noqa: F401  -- registers .hvplot
import numpy as np
import pandas as pd
import panel as pn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_src"))

from catraj.annotation import (  # noqa: E402
    DLT,
    IDW,
    RBF,
    CloughTocher,
    Kriging,
    Nearest,
    PiecewisePolynomial,
    Spline,
    TIN,
    annotate_points,
    build_series,
    decode_compass,
    describe_columns,
    evaluate_spatial,
    evaluate_temporal,
    parse_data_dictionary,
    to_surface,
)

pn.extension("tabulator", notifications=True, sizing_mode="stretch_width")
hv.extension("bokeh")

FIGURES = ROOT / "02_src" / "catraj" / "annotation"

INK, MUTED = "#0d1117", "#656d76"
ACCENT, GOOD, BAD, CONTEXT = "#0969da", "#1a7f37", "#cf222e", "#e16f24"

SPATIAL_METHODS = {
    "IDW": IDW,
    "Nearest": Nearest,
    "TIN": TIN,
    "CloughTocher": CloughTocher,
    "RBF": RBF,
    "Kriging": Kriging,
}
TEMPORAL_METHODS = {
    "DLT": DLT,
    "Piecewise polynomial": PiecewisePolynomial,
    "Spline": Spline,
}

TAB_OVERVIEW, TAB_DATA, TAB_SPATIAL, TAB_TEMPORAL, TAB_ANNOTATE = range(5)


class State:
    """Everything loaded or computed, shared across the tabs."""

    def __init__(self):
        self.trajectories = None
        self.stations = None
        self.spatial_table = None
        self.temporal_table = None
        self.annotated = None
        self.best_spatial = {}         # variable -> (method, mean rmse)
        self.best_temporal = {}        # variable -> (method, k, rmse)
        self.circular = {}             # variable -> bool
        self.labels = {}               # column -> human label


state = State()


# ----------------------------------------------------------------- helpers


def _notify(message, kind="info"):
    if pn.state.notifications:
        getattr(pn.state.notifications, kind)(message, duration=5000)


def _read(path):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(
            f"{Path(path).name} has no CRS. Interpolation weights distances, so "
            "the data needs a projected CRS in metres."
        )
    return gdf


def _numeric_columns(gdf):
    if gdf is None:
        return []
    return [c for c in gdf.columns
            if c != "geometry" and pd.api.types.is_numeric_dtype(gdf[c])]


#: Columns that locate an observation rather than measure anything. These are
#: never offered as context to annotate: interpolating a station id or a
#: timestamp is meaningless, and offering them invites the mistake. Matching is
#: exact -- `T` is air temperature, not a time column, and a substring rule
#: would wrongly drop it.
_NOT_CONTEXT = {
    "x", "y", "mx", "my", "geometry",
    "lon", "lat", "long", "longitude", "latitude", "easting", "northing",
    "time", "timestamp", "local_time", "unix_time", "epoch", "datetime", "date",
    "id", "fid", "oid", "objectid", "index", "id_station", "station_id", "station",
    "mmsi", "sourcemmsi", "traj_id",
}

#: Identifier-looking names, matched as whole words rather than substrings.
_ID_SUFFIXES = ("_id", "_fid", "_code", "_key")


def _context_candidates(columns, exclude=()):
    """Numeric columns that could plausibly be a measured context variable.

    Positions, times and identifiers are dropped: they place an observation
    rather than describe conditions, so interpolating them is meaningless.

    The rules are deliberately narrow. ``T`` is air temperature, not a time
    column; ``id_windDirection`` is a real measurement despite its prefix, so
    an ``id_`` prefix alone is not disqualifying -- only names that are
    identifiers outright.
    """
    out = []
    for column in columns:
        lowered = column.lower()
        if lowered in _NOT_CONTEXT or column in exclude:
            continue
        if lowered.endswith(_ID_SUFFIXES):
            continue
        out.append(column)
    return out


def _status(text, tone="muted"):
    colour = {"muted": MUTED, "good": GOOD, "bad": BAD}[tone]
    return pn.pane.HTML(
        f"<div style='color:{colour};font-size:13px;line-height:1.55'>{text}</div>"
    )


def _note(text):
    """A quiet explanatory aside beneath a control."""
    return pn.pane.HTML(
        f"<div style='color:{MUTED};font-size:12px;line-height:1.55;"
        f"margin:2px 0 4px 2px'>{text}</div>"
    )


def _heading(title, subtitle=""):
    sub = (f"<div style='color:{MUTED};font-size:13px;margin-top:3px'>"
           f"{subtitle}</div>") if subtitle else ""
    return pn.pane.HTML(
        f"<div style='margin:2px 0 12px'><div style='font-size:20px;"
        f"font-weight:600;color:{INK};letter-spacing:-.3px'>{title}</div>"
        f"{sub}</div>"
    )


def _card(*objects, title=None):
    head = [pn.pane.HTML(
        f"<div style='font-size:11px;font-weight:700;color:{MUTED};"
        f"letter-spacing:.09em;margin-bottom:2px'>{title.upper()}</div>"
    )] if title else []
    return pn.Column(
        *head, *objects,
        styles={"background": "#fafbfc", "border": "1px solid #e4e8ed",
                "border-radius": "10px", "padding": "14px 16px"},
    )


# ------------------------------------------------------- 0. overview tab

tabs = pn.Tabs(dynamic=False, sizing_mode="stretch_width")


def _goto(index):
    def handler(_=None):
        tabs.active = index
    return handler


def _svg(name):
    path = FIGURES / name
    if not path.exists():
        return _status(f"Figure not found: {name}", "bad")
    return pn.pane.SVG(str(path), sizing_mode="stretch_width")


def _stage_button(label, description, index, colour):
    """A stage card: a fixed-height description above a button.

    The height is pinned so cards whose descriptions wrap to different numbers
    of lines still put their buttons on one baseline.
    """
    button = pn.widgets.Button(name=label, button_type="light", width=210)
    button.on_click(_goto(index))
    return pn.Column(
        pn.pane.HTML(
            f"<div style='border-left:3px solid {colour};padding-left:10px;"
            f"height:62px;display:flex;align-items:flex-start'>"
            f"<div style='font-size:12px;color:{MUTED};line-height:1.5'>"
            f"{description}</div></div>",
            height=66, margin=(0, 0, 6, 0),
        ),
        button,
        width=230,
    )


figure_pick = pn.widgets.RadioButtonGroup(
    options=["Both stages", "Spatial detail", "Temporal detail"],
    value="Both stages", button_type="default",
)
figure_pane = pn.Column(_svg("annotation_master.svg"))


def _switch_figure(event):
    figure_pane[:] = [_svg({
        "Both stages": "annotation_master.svg",
        "Spatial detail": "spatial_flowchart.svg",
        "Temporal detail": "annotation_flowchart.svg",
    }[event.new])]


figure_pick.param.watch(_switch_figure, "value")

overview_tab = pn.Column(
    _heading(
        "Trajectory annotation",
        "Linking every trajectory point to the conditions the moving object "
        "passed through.",
    ),
    pn.pane.HTML(
        f"<div style='font-size:13.5px;line-height:1.7;color:{INK};"
        f"max-width:940px'>"
        "Movement and context are observed at different resolutions, and never "
        "in the same places. Context comes from scattered stations at coarse "
        "intervals; moving objects report position far more often, and away "
        "from every station. Two interpolations bridge that gap &mdash; first "
        "across space, then through time.<br><br>"
        "<span style='color:" + MUTED + "'>The sample data is maritime, but "
        "nothing here is: any trajectory and any measured context will "
        "do.</span></div>"
    ),
    pn.Spacer(height=14),
    pn.Row(
        _stage_button("1 - Data", "Load trajectories and context observations, "
                                  "and say which column is which.",
                      TAB_DATA, ACCENT),
        _stage_button("2 - Spatial", "Estimate context at the object's "
                                     "position, one epoch at a time.",
                      TAB_SPATIAL, CONTEXT),
        _stage_button("3 - Temporal", "Fit a curve through that series and read "
                                      "it at the exact timestamp.",
                      TAB_TEMPORAL, CONTEXT),
        _stage_button("4 - Annotate", "Apply the chosen methods and export the "
                                      "enriched trajectories.",
                      TAB_ANNOTATE, GOOD),
        pn.Spacer(),
    ),
    pn.Spacer(height=16),
    figure_pick,
    figure_pane,
    pn.pane.HTML(
        f"<div style='color:{MUTED};font-size:12px;margin-top:12px;"
        f"max-width:940px;line-height:1.6'>"
        "Mehri, Alesheikh &amp; Basiri (2023), <i>Ocean Engineering</i> 282, "
        "114916. The temporal stage is that paper's method; the spatial stage "
        "extends it &mdash; the paper generated per-epoch surfaces but placed "
        "the study of spatial interpolation outside its scope.</div>"
    ),
)


# ------------------------------------------------------------ 1. data tab

traj_path = pn.widgets.TextInput(
    name="Trajectory file", value="08_data/sample/AIS/AIS_DLT_UTM.shp")
station_path = pn.widgets.TextInput(
    name="Context station file",
    value="08_data/sample/Weather/Weather_Observation_UTM.shp")
traj_dict_path = pn.widgets.TextInput(
    name="Trajectory field descriptions (optional)",
    value="08_data/sample/AIS/AIS_parameters.txt")
station_dict_path = pn.widgets.TextInput(
    name="Context field descriptions (optional)",
    value="08_data/sample/Weather/Weather_parameters.txt")
load_button = pn.widgets.Button(name="Load", button_type="primary", width=120)
dictionary_status = pn.Column()

traj_time_col = pn.widgets.Select(name="Time column", options=[])
traj_id_col = pn.widgets.Select(name="Trajectory id column", options=[])
station_time_col = pn.widgets.Select(name="Epoch column", options=[])
value_cols = pn.widgets.MultiChoice(
    name="Context variables to annotate", options=[], value=[])
circular_cols = pn.widgets.MultiChoice(
    name="Which of those are bearings", options=[], value=[])
decode_cols = pn.widgets.MultiChoice(
    name="Which are compass-point codes (degrees = code x 22.5)",
    options=[], value=[])

data_status = pn.Column(_status("No data loaded yet."))
data_preview = pn.Column()
surface_epoch = pn.widgets.IntSlider(name="Epoch", start=0, end=1, value=0)


def load_data(_=None):
    try:
        def resolve(p):
            path = Path(p)
            return path if path.is_absolute() else ROOT / path

        traj = _read(resolve(traj_path.value))
        stations = _read(resolve(station_path.value))
    except Exception as exc:  # noqa: BLE001 -- surfaced to the user
        data_status[:] = [_status(f"Could not load: {exc}", "bad")]
        return

    state.trajectories, state.stations = traj, stations

    traj_numeric = _numeric_columns(traj)
    traj_time_col.options = {state.labels.get(c, c): c for c in traj_numeric}
    traj_time_col.value = next(
        (c for c in ("unix_time", "t", "time", "timestamp") if c in traj_numeric),
        traj_numeric[0] if traj_numeric else None)
    traj_id_col.options = [c for c in traj.columns if c != "geometry"]
    traj_id_col.value = next(
        (c for c in ("mmsi", "sourcemmsi", "traj_id", "id") if c in traj.columns),
        traj_id_col.options[0] if traj_id_col.options else None)

    # Optional data dictionaries. Column names like `Ff` or `id_windDir` say
    # nothing on their own; a description file turns them into readable labels.
    # Datasets whose names are already meaningful need none of this.
    traj_dict = (parse_data_dictionary(resolve(traj_dict_path.value))
                 if traj_dict_path.value.strip() else {})
    station_dict = (parse_data_dictionary(resolve(station_dict_path.value))
                    if station_dict_path.value.strip() else {})
    state.labels = {
        **describe_columns([c for c in traj.columns if c != "geometry"],
                           traj_dict),
        **describe_columns([c for c in stations.columns if c != "geometry"],
                           station_dict),
    }
    described = sum(1 for c, l in state.labels.items() if l != c)
    if traj_dict or station_dict:
        dictionary_status[:] = [_status(
            f"Described <b>{described}</b> of {len(state.labels)} columns from "
            f"{len(traj_dict) + len(station_dict)} dictionary entries.", "good")]
    elif traj_dict_path.value.strip() or station_dict_path.value.strip():
        dictionary_status[:] = [_status(
            "No field descriptions could be read from those paths - the column "
            "names are used as they are.", "bad")]
    else:
        dictionary_status[:] = [_note(
            "No description files given, so the raw column names are used. "
            "Supply them when the names are coded rather than meaningful.")]

    st_numeric = _numeric_columns(stations)
    station_time_col.options = st_numeric
    station_time_col.value = next(
        (c for c in ("unix_time", "local_time", "t", "time") if c in st_numeric),
        st_numeric[0] if st_numeric else None)
    # The epoch column is chosen separately, so it is not also a candidate.
    candidates = _context_candidates(st_numeric, exclude={station_time_col.value})
    labelled = {state.labels.get(c, c): c for c in candidates}
    value_cols.options = labelled
    default = [c for c in ("wind_speed", "air_temperature", "Ff", "T", "value")
               if c in candidates]
    value_cols.value = default or candidates[:1]
    circular_cols.options = labelled
    circular_cols.value = [c for c in circular_cols.value if c in candidates]
    decode_cols.options = labelled
    decode_cols.value = [c for c in decode_cols.value if c in candidates]
    station_time_col.options = {state.labels.get(c, c): c for c in st_numeric}

    same_crs = traj.crs == stations.crs
    if not same_crs:
        state.stations = stations.to_crs(traj.crs)
    crs_note = (f"both in {traj.crs.name}" if same_crs else
                f"CRS differed ({stations.crs.name}); stations reprojected to "
                f"{traj.crs.name}")
    n_traj = traj[traj_id_col.value].nunique() if traj_id_col.value else 0
    data_status[:] = [_status(
        f"<b>{len(traj):,}</b> trajectory points across <b>{n_traj}</b> "
        f"trajectories &nbsp;&middot;&nbsp; <b>{len(stations):,}</b> station "
        f"observations &nbsp;&middot;&nbsp; {crs_note}",
        "good" if same_crs else "muted")]

    epochs = np.sort(state.stations[station_time_col.value].astype(float).unique())
    surface_epoch.end = max(len(epochs) - 1, 1)
    surface_epoch.value = min(surface_epoch.value, surface_epoch.end)

    data_preview[:] = [pn.Row(
        pn.Column(
            pn.pane.HTML("<b style='font-size:13px'>Trajectories</b>"),
            pn.widgets.Tabulator(traj.drop(columns="geometry").head(50),
                                 height=230, disabled=True,
                                 layout="fit_data_table")),
        pn.Column(
            pn.pane.HTML("<b style='font-size:13px'>Context stations</b>"),
            pn.widgets.Tabulator(stations.drop(columns="geometry").head(50),
                                 height=230, disabled=True,
                                 layout="fit_data_table")),
    )]
    _describe_scope()
    _notify("Data loaded.", "success")


load_button.on_click(load_data)


def _station_frame(variable=None):
    """Station observations for one variable, as (x, y, t, value).

    Compass-point codes are decoded here, since interpolating the codes
    themselves is meaningless -- they are categorical labels, and a step from
    code 5 to code 13 is not a rotation.
    """
    variable = variable or _current_variable()
    st = state.stations
    values = st[variable].astype(float).to_numpy()
    if variable in decode_cols.value:
        values = decode_compass(values)
    return pd.DataFrame({
        "x": st.geometry.x, "y": st.geometry.y,
        "t": st[station_time_col.value].astype(float), "value": values})


def _current_variable():
    """The variable the comparison tabs are working on."""
    if study_variable.value:
        return study_variable.value
    return value_cols.value[0] if value_cols.value else None


def _is_circular(variable=None):
    return (variable or _current_variable()) in circular_cols.value


def _label(column):
    """The human label for a column, or the column name if none is known."""
    return state.labels.get(column, column)


def _sync_variable_pickers(*_):
    """Keep the per-tab variable selectors in step with the chosen set."""
    chosen = list(value_cols.value)
    labelled = {_label(c): c for c in chosen}
    for picker in (study_variable, temporal_variable):
        picker.options = labelled
        if picker.value not in chosen:
            picker.value = chosen[0] if chosen else None
    circular_cols.options = chosen or circular_cols.options
    decode_cols.options = chosen or decode_cols.options
    _describe_scope()


data_tab = pn.Column(
    _heading("Load your data",
             "Two files: the trajectories to annotate, and the context "
             "observations to annotate them from."),
    _card(
        pn.Row(traj_path, station_path,
               pn.Column(pn.Spacer(height=18), load_button)),
        pn.Row(traj_dict_path, station_dict_path),
        _note("Field descriptions are <b>optional</b>. Supply them when the "
              "column names are coded rather than meaningful &mdash; "
              "<code>Ff</code> reads as \"Mean wind speed (meters per "
              "seconds)\" with the weather dictionary, and as "
              "<code>Ff</code> without it."),
        dictionary_status,
        _note("Both files need a <b>projected CRS in metres</b>. Interpolation weights "
              "distances, and degrees are not a metric space &mdash; a degree of "
              "longitude at 48&deg;N spans about 74 km against 111 km for a "
              "degree of latitude, so distances measured in degrees are "
              "anisotropic."),
        data_status,
        title="Files",
    ),
    pn.Spacer(height=10),
    _card(
        pn.Row(
            pn.Column(pn.pane.HTML(
                f"<div style='font-size:11px;font-weight:700;color:{MUTED};"
                "letter-spacing:.08em'>TRAJECTORIES</div>"),
                traj_time_col, traj_id_col),
            pn.Column(pn.pane.HTML(
                f"<div style='font-size:11px;font-weight:700;color:{MUTED};"
                "letter-spacing:.08em'>CONTEXT</div>"),
                station_time_col),
        ),
        pn.layout.Divider(),
        value_cols,
        _note("Pick every variable you want annotated. Each is compared and "
              "annotated independently, so temperature can use one "
              "interpolator and wind direction another. Coordinates, "
              "timestamps and identifiers are not listed &mdash; they place an "
              "observation rather than measure anything."),
        circular_cols,
        decode_cols,
        _note("Nothing is inferred from a column name. Wind and wave "
              "<i>direction</i> are bearings: interpolated as ordinary numbers, "
              "350&deg; and 10&deg; average to 180&deg; &mdash; the opposite "
              "heading. Listing a variable above interpolates it on the unit "
              "circle instead."),
        title="Column mapping",
    ),
    pn.Spacer(height=10),
    data_preview,
)


# --------------------------------------------------------- 2. spatial tab

study_variable = pn.widgets.Select(name="Variable", options=[])
spatial_methods = pn.widgets.MultiChoice(
    name="Methods to compare", value=["IDW", "Nearest", "TIN", "RBF"],
    options=list(SPATIAL_METHODS))
run_spatial = pn.widgets.Button(
    name="Run cross-validation", button_type="primary", width=210)
spatial_out = pn.Column(_status("Load data, then run the comparison."))


def do_spatial(_=None):
    if state.stations is None:
        _notify("Load data first.", "warning")
        return
    spatial_out[:] = [pn.indicators.LoadingSpinner(value=True, size=26)]
    try:
        variable = _current_variable()
        if variable is None:
            spatial_out[:] = [_status("Choose a context variable first.", "bad")]
            return
        frame = _station_frame(variable)
        table = evaluate_spatial(
            frame.x, frame.y, frame.value, frame.t,
            methods=[k.lower().replace("cloughtocher", "clough_tocher")
                     for k in spatial_methods.value],
            circular=_is_circular(variable))
        state.spatial_table = table
        df = table.to_frame()

        winner = table.select()
        state.best_spatial[variable] = (
            winner, float(df[df.method == winner].rmse.mean()))
        _sync_recommendations()

        ranked = (df.groupby("method")[["rmse", "mae", "me", "max_ae",
                                        "unreachable"]]
                  .mean().sort_values("rmse").round(3).reset_index())
        df["time"] = pd.to_datetime(df.epoch, unit="s")
        n_stations = int(df.n_stations.max())

        spatial_out[:] = [
            _status(f"<b>{_label(variable)}</b> &mdash; best by mean RMSE: "
                    f"<b>{winner}</b> "
                    f"({state.best_spatial[variable][1]:.4g}). Carried to the "
                    f"Annotate tab.", "good"),
            pn.widgets.Tabulator(ranked, disabled=True, height=210,
                                 layout="fit_data_table"),
            _note(f"Each fit used {n_stations - 1} of {n_stations} stations. "
                  "<b>Read <code>unreachable</code> beside RMSE</b>: a held-out "
                  "station outside the hull of the others cannot be predicted "
                  "by the triangulation methods at all, so a method can score "
                  "well while failing exactly where the object is most "
                  "likely to be."),
            pn.pane.HoloViews(
                df.hvplot.line(x="time", y="rmse", by="method", muted_alpha=0.1,
                               height=300, responsive=True, ylabel="RMSE",
                               title="error by epoch")
                .opts(legend_position="right")),
            _note("Click a method in the legend to mute it."),
        ]
    except Exception:  # noqa: BLE001
        spatial_out[:] = [_status(f"<pre>{traceback.format_exc()}</pre>", "bad")]


run_spatial.on_click(do_spatial)

surface_method = pn.widgets.Select(
    name="Method", options=list(SPATIAL_METHODS), value="IDW")
draw_surface = pn.widgets.Button(name="Draw", button_type="primary", width=120)
surface_out = pn.Column()


def do_surface(_=None):
    if state.stations is None:
        _notify("Load data first.", "warning")
        return
    surface_out[:] = [pn.indicators.LoadingSpinner(value=True, size=26)]
    try:
        variable = _current_variable()
        frame = _station_frame(variable)
        epochs = np.sort(frame.t.unique())
        surface_epoch.end = max(len(epochs) - 1, 1)
        idx = min(surface_epoch.value, len(epochs) - 1)
        rows = frame[frame.t == epochs[idx]]

        method = SPATIAL_METHODS[surface_method.value](
            circular=_is_circular(variable))
        fitted = method.fit(rows.x, rows.y, rows.value)

        pad = 0.15 * max(np.ptp(rows.x), np.ptp(rows.y))
        bounds = (rows.x.min() - pad, rows.y.min() - pad,
                  rows.x.max() + pad, rows.y.max() + pad)
        grid, xs, ys = to_surface(fitted, bounds, (bounds[2] - bounds[0]) / 90)
        covered = 100 * np.isfinite(grid).mean()

        image = hv.Image((xs, ys, grid), kdims=["x", "y"], vdims="value").opts(
            cmap="viridis", colorbar=True, responsive=True, height=430,
            title=f"{variable} - {surface_method.value} at "
                  f"{pd.to_datetime(epochs[idx], unit='s')} UTC")
        points = rows.hvplot.points(x="x", y="y", c="value", cmap="viridis",
                                    size=110, line_color="white",
                                    hover_cols=["value"])
        surface_out[:] = [
            pn.pane.HoloViews(image * points),
            _note(f"<b>{covered:.0f}%</b> of the area covered. Blank regions lie "
                  "outside the convex hull of the stations; the triangulation "
                  "methods report nothing there rather than inventing a value."),
        ]
    except Exception:  # noqa: BLE001
        surface_out[:] = [_status(f"<pre>{traceback.format_exc()}</pre>", "bad")]


draw_surface.on_click(do_surface)

spatial_tab = pn.Column(
    _heading("Stage 1 - spatial interpolation",
             "Context is measured at stations, never where the moving object "
             "is."),
    _card(
        pn.pane.HTML(
            f"<div style='font-size:13px;line-height:1.65;color:{INK}'>"
            "Each method is scored by <b>leave-one-station-out</b>: hold out one "
            "station, predict its value from the rest, compare with what it "
            "actually observed. Repeat for every station, at every epoch. This "
            "is the spatial analogue of k-fold, with stations rather than time "
            "as the folds.</div>"),
        pn.Row(study_variable, spatial_methods,
               pn.Column(pn.Spacer(height=18), run_spatial)),
        _note("Each variable is compared separately - a method that suits wind "
              "speed need not suit wind direction."),
        title="Compare methods",
    ),
    pn.Spacer(height=8),
    spatial_out,
    pn.Spacer(height=10),
    _card(
        pn.Row(surface_method, surface_epoch,
               pn.Column(pn.Spacer(height=18), draw_surface)),
        _note("Annotation samples at points and never builds a raster. This view "
              "exists only to show how the methods differ."),
        title="Inspect a surface",
    ),
    surface_out,
)


# -------------------------------------------------------- 3. temporal tab

temporal_variable = pn.widgets.Select(name="Variable", options=[])
temporal_where = pn.widgets.Select(
    name="Series to fit",
    options=["A station's own observations",
             "The interpolated series at a trajectory"],
    value="A station's own observations")
temporal_degrees = pn.widgets.IntRangeSlider(
    name="Piecewise polynomial degrees", start=2, end=7, value=(2, 5), step=1)
k_range = pn.widgets.IntRangeSlider(
    name="Fold counts (k)", start=2, end=10, value=(2, 10), step=1)
run_temporal = pn.widgets.Button(
    name="Run k-fold", button_type="primary", width=180)
temporal_out = pn.Column(_status("Load data, then run the comparison."))


def _temporal_series(variable):
    """The (t, v) series the comparison is run on, and a label for it."""
    frame = _station_frame(variable)
    if (temporal_where.value.startswith("A station")
            or state.trajectories is None or not vessel_pick.options):
        first = state.stations.iloc[0]
        rows = frame[(frame.x == first.geometry.x)
                     & (frame.y == first.geometry.y)].sort_values("t")
        return rows.t.to_numpy(), rows.value.to_numpy(), "one station's own observations"

    traj = state.trajectories
    chosen = traj[traj[traj_id_col.value].astype(str) == str(vessel_pick.value)]
    if chosen.empty:
        chosen = traj.head(1)
    point = chosen.geometry.iloc[len(chosen) // 2]
    times, values = build_series(
        frame, point.x, point.y,
        spatial_method=SPATIAL_METHODS[use_spatial.value](
            circular=_is_circular(variable)),
        circular=_is_circular(variable))
    return times, values, (f"the interpolated series at trajectory "
                           f"{vessel_pick.value}")


def do_temporal(_=None):
    if state.stations is None:
        _notify("Load data first.", "warning")
        return
    temporal_out[:] = [pn.indicators.LoadingSpinner(value=True, size=26)]
    try:
        variable = temporal_variable.value or _current_variable()
        if variable is None:
            temporal_out[:] = [_status("Choose a context variable first.", "bad")]
            return
        t, v, where = _temporal_series(variable)
        ok = np.isfinite(t) & np.isfinite(v)
        t, v = t[ok], v[ok]

        candidates = [("dlt", DLT, {})]
        candidates += [(f"pp_deg{d}", PiecewisePolynomial, {"degree": d})
                       for d in range(temporal_degrees.value[0],
                                      temporal_degrees.value[1] + 1)]
        candidates.append(("spline", Spline, {}))

        table = evaluate_temporal(
            t, v, candidates=candidates,
            ks=range(k_range.value[0], k_range.value[1] + 1),
            circular=_is_circular(variable))
        state.temporal_table = table
        tdf = table.to_frame()

        by_rmse, by_mae = table.best_row("rmse"), table.best_row("mae")
        state.best_temporal[variable] = (
            by_rmse.method, by_rmse.k, by_rmse.metrics.rmse)
        _sync_recommendations()

        dense = np.linspace(t.min(), t.max(), 500)
        curves = pd.concat([
            pd.DataFrame({
                "time": pd.to_datetime(dense, unit="s"),
                "value": factory(circular=_is_circular(variable), **kw)
                         .fit(t, v)(dense),
                "method": name})
            for name, factory, kw in candidates])
        observed = pd.DataFrame({"time": pd.to_datetime(t, unit="s"), "value": v})

        agree = (by_rmse.method, by_rmse.k) == (by_mae.method, by_mae.k)
        verdict = ("Both rules agree." if agree else
                   "<b>The two rules disagree.</b> Which to trust is a judgement "
                   "about whether rare large errors matter more than typical "
                   "ones &mdash; RMSE weights the large ones, MAE treats all "
                   "equally. The app does not decide it for you.")

        temporal_out[:] = [
            _status(f"<b>{_label(variable)}</b> - fitted to {len(t)} "
                    f"observations "
                    f"from {where}.<br>"
                    f"Best by RMSE: <b>{by_rmse.method}</b> (k={by_rmse.k}) "
                    f"&nbsp;&middot;&nbsp; best by MAE: <b>{by_mae.method}</b> "
                    f"(k={by_mae.k})<br>{verdict}",
                    "good" if agree else "muted"),
            pn.pane.HoloViews(
                curves.hvplot.line(x="time", y="value", by="method",
                                   muted_alpha=0.15, height=330,
                                   responsive=True, title="fitted curves")
                .opts(legend_position="right")
                * observed.hvplot.scatter(x="time", y="value", color=CONTEXT,
                                          size=60, line_color="white")),
            _note("Zoom into a single gap to see where the methods diverge. The "
                  "piecewise polynomial passes <i>near</i> its observations, not "
                  "through them &mdash; it is a least-squares fit. Only DLT and "
                  "the spline are interpolating."),
            pn.pane.HoloViews(
                tdf.hvplot.heatmap(x="k", y="method", C="rmse", cmap="viridis_r",
                                   height=280, responsive=True,
                                   title="every method at every fold count - "
                                         "darker is better")
                .opts(xrotation=0)),
        ]
    except Exception:  # noqa: BLE001
        temporal_out[:] = [_status(f"<pre>{traceback.format_exc()}</pre>", "bad")]


run_temporal.on_click(do_temporal)

temporal_tab = pn.Column(
    _heading("Stage 2 - temporal interpolation",
             "A curve through the epoch series supplies a value at the exact "
             "timestamp."),
    _card(
        pn.pane.HTML(
            f"<div style='font-size:13px;line-height:1.65;color:{INK}'>"
            "Observations are partitioned randomly into k sets; each fold is "
            "held out in turn and scored, and the reported metrics are the mean "
            "over all k folds. <b>The first and last observation are always kept "
            "in training</b>, so no method is ever asked to extrapolate.</div>"),
        pn.Row(temporal_variable, temporal_where),
        pn.Row(temporal_degrees, k_range),
        pn.Row(run_temporal, pn.Spacer()),
        _note("Sweeping k varies both the fold structure and the number of "
              "fitting points, since training size grows with k."),
        title="Compare methods",
    ),
    pn.Spacer(height=8),
    temporal_out,
)


# -------------------------------------------------------- 4. annotate tab

recommendation = pn.Column()
use_recommended = pn.widgets.Button(
    name="Use recommended", button_type="success", width=190, disabled=True)
use_spatial = pn.widgets.Select(
    name="Spatial method", options=list(SPATIAL_METHODS), value="IDW")
use_temporal = pn.widgets.Select(
    name="Temporal method", options=list(TEMPORAL_METHODS), value="Spline")
pp_degree = pn.widgets.IntSlider(
    name="Polynomial degree (piecewise only)", start=2, end=7, value=2)
per_variable_methods = pn.widgets.Checkbox(
    name="Use each variable's own cross-validated best, where one exists",
    value=True)

scope = pn.widgets.RadioButtonGroup(
    options=["One trajectory", "All trajectories"],
    value="One trajectory", button_type="primary")
vessel_pick = pn.widgets.Select(name="Trajectory", options=[])
max_trajectories = pn.widgets.IntInput(
    name="Limit (0 = all)", value=0, start=0, step=10, width=150)
run_annotate = pn.widgets.Button(name="Annotate", button_type="primary", width=160)
progress = pn.indicators.Progress(value=0, max=100, visible=False)

export_dir = pn.widgets.TextInput(
    name="Folder", value=str(ROOT / "10_results" / "03_app_exports"))
export_name = pn.widgets.TextInput(name="File name", value="annotated", width=260)
export_button = pn.widgets.Button(name="Export", button_type="primary", width=120)
export_status = pn.Column()
annotate_out = pn.Column(_status("Load data, then annotate."))
scope_note = pn.Column()

# A modal-style confirmation shown when a comparison has been skipped.
proceed_anyway = pn.widgets.Button(
    name="Annotate anyway", button_type="warning", width=180)
go_spatial = pn.widgets.Button(name="Go to Spatial", width=150)
go_temporal = pn.widgets.Button(name="Go to Temporal", width=150)
guard = pn.Column(visible=False)

go_spatial.on_click(_goto(TAB_SPATIAL))
go_temporal.on_click(_goto(TAB_TEMPORAL))


def _describe_scope(*_):
    """Say how many points the next run will annotate, before it starts."""
    if state.trajectories is None or not traj_id_col.value:
        scope_note[:] = [_note("Load data to see how many points this covers.")]
        return

    traj = state.trajectories
    n_vars = max(len(value_cols.value), 1)
    if scope.value == "One trajectory":
        n_points = int((traj[traj_id_col.value].astype(str)
                        == str(vessel_pick.value)).sum())
        scope_note[:] = [_note(
            f"<b>{n_points:,}</b> positions in trajectory {vessel_pick.value}, "
            f"x <b>{n_vars}</b> variable{'s' if n_vars > 1 else ''}.")]
        return

    counts = traj[traj_id_col.value].value_counts()
    if max_trajectories.value:
        counts = counts.iloc[: max_trajectories.value]
    scope_note[:] = [_note(
        f"<b>{int(counts.sum()):,}</b> positions across <b>{len(counts)}</b> "
        f"trajectories, x <b>{n_vars}</b> "
        f"variable{'s' if n_vars > 1 else ''}.")]


def _sync_recommendations():
    """Surface what cross-validation chose, per variable."""
    chosen = list(value_cols.value)
    rows = []
    for variable in chosen:
        sp = state.best_spatial.get(variable)
        tp = state.best_temporal.get(variable)
        cell = lambda v: (v if v else
                          f"<span style='color:{MUTED}'>not compared</span>")
        rows.append(
            f"<tr><td style='padding:3px 16px 3px 0'><b>{_label(variable)}"
            f"</b></td>"
            f"<td style='padding:3px 16px 3px 0'>"
            f"{cell(sp[0] if sp else None)}</td>"
            f"<td style='padding:3px 16px 3px 0'>"
            f"{cell(f'{tp[0]} (k={tp[1]})' if tp else None)}</td></tr>")

    if not rows:
        recommendation[:] = [_status(
            "Choose context variables on the Data tab first.")]
        use_recommended.disabled = True
        return

    any_result = bool(state.best_spatial or state.best_temporal)
    recommendation[:] = [pn.pane.HTML(
        "<div style='background:#ddf4ff;border:1px solid #54aeff;"
        "border-radius:10px;padding:13px 16px;font-size:13px'>"
        "<div style='font-weight:600;margin-bottom:7px'>"
        "Cross-validated best, per variable</div>"
        "<table style='font-size:13px'>"
        f"<tr><td style='color:{MUTED};padding-right:16px'>variable</td>"
        f"<td style='color:{MUTED};padding-right:16px'>spatial</td>"
        f"<td style='color:{MUTED}'>temporal</td></tr>"
        f"{''.join(rows)}</table></div>")]
    use_recommended.disabled = not any_result


def apply_recommended(_=None):
    """Set the fallback selectors from the variable being studied."""
    variable = _current_variable()
    sp = state.best_spatial.get(variable)
    tp = state.best_temporal.get(variable)
    if sp:
        pretty = {k.lower(): k for k in SPATIAL_METHODS}
        name = sp[0].replace("_", "").lower()
        if name in pretty:
            use_spatial.value = pretty[name]
    if tp:
        name = tp[0]
        if name.startswith("pp_deg"):
            use_temporal.value = "Piecewise polynomial"
            pp_degree.value = int(name.replace("pp_deg", ""))
        elif name in ("dlt", "spline"):
            use_temporal.value = {"dlt": "DLT", "spline": "Spline"}[name]
    _notify(f"Methods set from the best for {variable}.", "success")


use_recommended.on_click(apply_recommended)


def refresh_vessels(*_):
    if state.trajectories is None or not traj_id_col.value:
        return
    counts = state.trajectories[traj_id_col.value].value_counts()
    vessel_pick.options = [str(v) for v in counts.index[:500]]
    if vessel_pick.options:
        vessel_pick.value = vessel_pick.options[0]
    _describe_scope()


traj_id_col.param.watch(refresh_vessels, "value")
value_cols.param.watch(_sync_variable_pickers, "value")


def _toggle_scope(event):
    single = event.new == "One trajectory"
    vessel_pick.disabled = not single
    max_trajectories.disabled = single
    _describe_scope()


scope.param.watch(_toggle_scope, "value")
vessel_pick.param.watch(_describe_scope, "value")
max_trajectories.param.watch(_describe_scope, "value")
max_trajectories.disabled = True


def _methods_for(variable):
    """The (spatial, temporal) pair to use for one variable.

    Each variable's own cross-validated winner is preferred; the selectors are
    the fallback for variables that were never compared.
    """
    circular = _is_circular(variable)
    spatial_cls = SPATIAL_METHODS[use_spatial.value]
    temporal_cls = TEMPORAL_METHODS[use_temporal.value]
    kwargs = {"degree": pp_degree.value} if temporal_cls is PiecewisePolynomial else {}

    if per_variable_methods.value:
        sp = state.best_spatial.get(variable)
        if sp:
            lookup = {k.lower(): v for k, v in SPATIAL_METHODS.items()}
            spatial_cls = lookup.get(sp[0].replace("_", "").lower(), spatial_cls)
        tp = state.best_temporal.get(variable)
        if tp:
            name = tp[0]
            if name.startswith("pp_deg"):
                temporal_cls = PiecewisePolynomial
                kwargs = {"degree": int(name.replace("pp_deg", ""))}
            elif name == "dlt":
                temporal_cls, kwargs = DLT, {}
            elif name == "spline":
                temporal_cls, kwargs = Spline, {}

    return (spatial_cls(circular=circular),
            temporal_cls(circular=circular, **kwargs))


def _missing_comparisons():
    """Variables whose spatial or temporal comparison has not been run."""
    missing = []
    for variable in value_cols.value:
        gaps = []
        if variable not in state.best_spatial:
            gaps.append("spatial")
        if variable not in state.best_temporal:
            gaps.append("temporal")
        if gaps:
            missing.append((variable, gaps))
    return missing


def request_annotate(_=None):
    """Warn about skipped comparisons, but never block: the user may already
    know which methods suit their data, and re-running is expensive."""
    if state.trajectories is None:
        _notify("Load data first.", "warning")
        return
    if not value_cols.value:
        _notify("Choose at least one context variable on the Data tab.",
                "warning")
        return

    missing = _missing_comparisons()
    if not missing:
        guard.visible = False
        do_annotate()
        return

    lines = "".join(
        f"<li><b>{v}</b> &mdash; no {' or '.join(g)} comparison</li>"
        for v, g in missing)
    guard[:] = [
        pn.pane.HTML(
            "<div style='background:#fff8c5;border:1px solid #d4a72c;"
            "border-radius:10px;padding:13px 16px;font-size:13px'>"
            "<b>Some comparisons have not been run.</b>"
            f"<ul style='margin:8px 0 8px 18px;padding:0'>{lines}</ul>"
            f"<span style='color:{MUTED}'>Those variables will use the methods "
            "selected above. If you already know which methods suit your data, "
            "that is fine &mdash; carry on.</span></div>"),
        pn.Row(proceed_anyway, go_spatial, go_temporal, pn.Spacer()),
    ]
    guard.visible = True


def do_annotate(_=None):
    guard.visible = False
    annotate_out[:] = [pn.indicators.LoadingSpinner(value=True, size=26)]
    try:
        traj = state.trajectories
        variables = list(value_cols.value)

        if scope.value == "One trajectory":
            subset = traj[traj[traj_id_col.value].astype(str)
                          == str(vessel_pick.value)]
        else:
            ids = traj[traj_id_col.value].value_counts().index
            if max_trajectories.value:
                ids = ids[: max_trajectories.value]
            subset = traj[traj[traj_id_col.value].isin(ids)]

        # Every point the user supplied is annotated. Cost scales with the
        # number of distinct positions, so one batched call per variable is far
        # cheaper than looping trajectory by trajectory.
        subset = subset.sort_values(
            [traj_id_col.value, traj_time_col.value]).copy()
        if subset.empty:
            annotate_out[:] = [_status("Nothing to annotate.", "bad")]
            return

        xs = subset.geometry.x.to_numpy()
        ys = subset.geometry.y.to_numpy()
        ts = subset[traj_time_col.value].to_numpy(float)

        progress.visible = True
        progress.max = len(variables)
        progress.value = 0

        # A trajectory file may already carry a column of the same name, and
        # writing a duplicate breaks GeoPackage export, so annotated columns
        # are named apart from whatever the source already holds.  The check
        # ignores case: GeoPackage is SQLite, whose column names are
        # case-insensitive, so the weather variable `T` (air temperature)
        # collides with the AIS timestamp `t`.
        existing = {c.lower() for c in subset.columns}
        used, columns = [], {}
        for n, variable in enumerate(variables, start=1):
            column = variable
            if column.lower() in existing:
                column = f"{variable}_ctx"
                while column.lower() in existing:
                    column += "_"
            columns[variable] = column
            existing.add(column.lower())

            spatial, temporal = _methods_for(variable)
            subset[column] = annotate_points(
                _station_frame(variable), xs, ys, ts,
                spatial_method=spatial, temporal_method=temporal,
                circular=_is_circular(variable))
            used.append((column, type(spatial).__name__,
                         type(temporal).__name__))
            progress.value = n

        variables = list(columns.values())

        progress.visible = False
        subset["time"] = pd.to_datetime(subset[traj_time_col.value], unit="s")
        state.annotated = subset

        n_traj = subset[traj_id_col.value].nunique()
        merc = subset.to_crs(3857)
        plot_df = subset.assign(mx=merc.geometry.x, my=merc.geometry.y)

        method_rows = "".join(
            f"<tr><td style='padding:2px 14px 2px 0'><b>{v}</b></td>"
            f"<td style='padding:2px 14px 2px 0'>{sp}</td>"
            f"<td>{tp}</td></tr>" for v, sp, tp in used)
        totals = {v: int(subset[v].notna().sum()) for v in variables}
        complete = all(n == len(subset) for n in totals.values())

        panels = [
            _status(
                f"Annotated <b>{len(subset):,}</b> positions across "
                f"<b>{n_traj}</b> trajector{'y' if n_traj == 1 else 'ies'} "
                f"with <b>{len(variables)}</b> "
                f"variable{'s' if len(variables) > 1 else ''}.",
                "good" if complete else "bad"),
            pn.pane.HTML(
                f"<table style='font-size:12.5px;color:{MUTED}'>"
                f"<tr><td style='padding-right:14px'>variable</td>"
                f"<td style='padding-right:14px'>spatial</td>"
                f"<td>temporal</td></tr>{method_rows}</table>"),
        ]

        show = pn.widgets.Select(
            name="Show", options=variables, value=variables[0], width=220)
        view = pn.Column()

        def _redraw(*_):
            variable = show.value
            view[:] = [
                pn.pane.HoloViews(
                    plot_df.hvplot.points(
                        x="mx", y="my", c=variable, cmap="viridis", size=16,
                        tiles="OSM", data_aspect=1, height=460,
                        responsive=True, xaxis=None, yaxis=None,
                        hover_cols=[traj_id_col.value],
                        title=f"{variable} along the annotated positions")),
            ]
            if n_traj == 1:
                view.append(pn.pane.HoloViews(
                    plot_df.hvplot.line(x="time", y=variable, color=GOOD,
                                        height=280, responsive=True,
                                        title=f"{variable} along the path")))

        show.param.watch(_redraw, "value")
        _redraw()
        panels += [show, view]

        if n_traj > 1:
            summary = (subset.groupby(traj_id_col.value)[variables]
                       .mean().round(3).reset_index().head(300))
            panels += [
                pn.pane.HTML("<b style='font-size:13px'>Per-trajectory "
                             "means</b>"),
                pn.widgets.Tabulator(summary, disabled=True, height=260,
                                     layout="fit_data_table"),
            ]
        panels.append(pn.widgets.Tabulator(
            subset.drop(columns="geometry").head(100), disabled=True,
            height=240, layout="fit_data_table"))

        annotate_out[:] = panels
        _notify(f"Annotated {len(variables)} variable(s).", "success")
    except Exception:  # noqa: BLE001
        progress.visible = False
        annotate_out[:] = [_status(f"<pre>{traceback.format_exc()}</pre>", "bad")]


def do_export(_=None):
    if state.annotated is None:
        _notify("Nothing to export - annotate first.", "warning")
        return

    stem = (export_name.value or "annotated").strip()
    folder = Path(export_dir.value.strip() or (ROOT / "10_results" / "03_app_exports"))
    if not folder.is_absolute():
        folder = ROOT / folder

    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        export_status[:] = [_status(f"Could not create {folder}: {exc}", "bad")]
        return

    try:
        written = []
        state.annotated.to_file(folder / f"{stem}.gpkg", driver="GPKG")
        written.append(f"{stem}.gpkg")
        state.annotated.drop(columns="geometry").to_csv(
            folder / f"{stem}.csv", index=False)
        written.append(f"{stem}.csv")
        if state.spatial_table is not None:
            state.spatial_table.to_frame().to_csv(
                folder / f"{stem}_spatial_scores.csv", index=False)
            written.append(f"{stem}_spatial_scores.csv")
        if state.temporal_table is not None:
            state.temporal_table.to_frame().to_csv(
                folder / f"{stem}_temporal_scores.csv", index=False)
            written.append(f"{stem}_temporal_scores.csv")
    except Exception as exc:  # noqa: BLE001 -- shown to the user
        export_status[:] = [_status(f"Export failed: {exc}", "bad")]
        return

    listed = "".join(f"<li><code>{name}</code></li>" for name in written)
    export_status[:] = [pn.pane.HTML(
        f"<div style='font-size:13px;color:{GOOD}'>Wrote {len(written)} files "
        f"to <code>{folder}</code>"
        f"<ul style='margin:6px 0 0 18px;padding:0;color:{MUTED}'>{listed}"
        f"</ul></div>")]
    _notify(f"Wrote {len(written)} files.", "success")


run_annotate.on_click(request_annotate)
proceed_anyway.on_click(lambda _=None: do_annotate())
export_button.on_click(do_export)
_sync_recommendations()
_describe_scope()

annotate_tab = pn.Column(
    _heading("Annotate",
             "Apply the chosen methods and write the context onto the "
             "trajectories."),
    recommendation,
    pn.Spacer(height=8),
    _card(
        per_variable_methods,
        _note("With this ticked, every variable uses whatever its own "
              "comparison chose. The selectors below are the fallback for "
              "variables that were never compared."),
        pn.Row(use_recommended, pn.Spacer()),
        pn.Row(use_spatial, use_temporal, pp_degree),
        title="Methods",
    ),
    pn.Spacer(height=10),
    _card(
        scope,
        pn.Row(vessel_pick, max_trajectories),
        scope_note,
        _note("Every point in the selected trajectories is annotated, for every "
              "chosen variable."),
        pn.Row(run_annotate, pn.Spacer()),
        progress,
        title="Scope",
    ),
    guard,
    pn.Spacer(height=8),
    annotate_out,
    pn.Spacer(height=10),
    _card(
        export_dir,
        pn.Row(export_name, pn.Column(pn.Spacer(height=18), export_button)),
        _note("Writes a GeoPackage and a CSV, plus the cross-validation scores "
              "behind the choice. A relative folder is taken from the project "
              "root; the folder is created if it does not exist."),
        export_status,
        title="Export",
    ),
)


# ------------------------------------------------------------------ layout

tabs.extend([
    ("Overview", overview_tab),
    ("1 - Data", data_tab),
    ("2 - Spatial", spatial_tab),
    ("3 - Temporal", temporal_tab),
    ("4 - Annotate", annotate_tab),
])

template = pn.template.FastListTemplate(
    title="catraj - trajectory annotation",
    accent_base_color=ACCENT,
    header_background=ACCENT,
    main=[tabs],
    sidebar=[pn.pane.HTML(
        f"<div style='font-size:13px;line-height:1.65;color:{INK}'>"
        "<div style='font-weight:600;font-size:14px;margin-bottom:8px'>"
        "The method</div>"
        "Movement is observed far more often than context, and never in the "
        "same places. Two interpolations bridge the gap. Any moving object "
        "will do &mdash; the sample data is maritime, the method is not."
        "<div style='margin:14px 0 4px'><b>1&nbsp; Spatial</b><br>"
        f"<span style='color:{MUTED}'>across stations, one epoch at a time"
        "</span></div>"
        "<div style='margin:10px 0 4px'><b>2&nbsp; Temporal</b><br>"
        f"<span style='color:{MUTED}'>through time, at that position</span>"
        "</div>"
        "<hr style='border:none;border-top:1px solid #e4e8ed;margin:16px 0'>"
        f"<div style='color:{MUTED};font-size:12px;line-height:1.6'>"
        "Mehri, Alesheikh &amp; Basiri (2023)<br>"
        "<i>Ocean Engineering</i> 282, 114916</div></div>")],
    sidebar_width=280,
)

template.servable()
