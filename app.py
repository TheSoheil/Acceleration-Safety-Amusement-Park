import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import butter, lfilter

# --- Set Page Layout to Wide ---
st.set_page_config(layout="wide")


# --- CSS Styling ---
st.markdown(
    """
    <style>
        /* Default background and text color */
        .stApp {
            background-color: #FFFFFF !important;  /* White background */
            color: #333333 !important;  /* Dark text for contrast */
        }

        /* Restore default container behavior */
        .stContainer {
            margin: 0 !important;
            padding: 0 !important;
        }

        /* Default button styling */
        div.stButton > button:first-child {
            background-color: #2196F3 !important;  /* Blue button */
            color: white !important;
            border-radius: 8px !important;
            height: 3em !important;
            font-size: 16px !important;
            border: none !important;
        }

        div.stButton > button:first-child:hover {
            background-color: #1976D2 !important;  /* Darker blue for hover effect */
        }

        /* Default headers styling */
        h1 {
            color: #2196F3 !important;
            font-size: 36px !important;
            text-align: center !important;
            font-weight: bold !important;
        }

        /* Default text for labels and inputs */
        .stSelectbox label, .stNumberInput label {
            color: #333333 !important;
            font-weight: bold !important;
        }

        /* Default success/error messages styling */
        .stSuccess {
            background-color: #388E3C !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 10px !important;
            font-weight: bold !important;
        }

        .stError {
            background-color: #D32F2F !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 10px !important;
            font-weight: bold !important;
        }

        /* Default markdown styling */
        .stMarkdown p {
            color: #333333 !important;
        }

        /* Success/Error Button */
        .stSuccess, .stError {
            font-size: 16px !important;
            text-align: center !important;
            border-radius: 8px !important;
        }
        /* Table styling */
        .stDataFrame {
            width: 100% !important;  /* Ensure the table takes up full width */
            table-layout: auto !important;  /* Allow columns to adjust based on content */
        }
        
        table {
            width: 100% !important;  /* Make sure the table is full width */
            table-layout: auto !important;  /* Allow table columns to adjust dynamically */
            color: #333333 !important;  /* Set text color for table content */
        }
        
        th, td {
            padding: 8px !important;
            text-align: left !important;
            border: 1px solid #ddd !important;  /* Border styling */
        }
        
        .stDataFrame > div > div {
            background-color: #FFFFFF !important;  /* Ensure background color for tables */
        }
        
        .stDataFrame table {
            width: 100% !important;  /* Full width for tables */
            font-size: 14px !important;  /* Set font size */
            word-wrap: break-word !important;  /* Prevent word overflow */
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# --- ISO 17842-1:2023 Thresholds ---
THRESHOLDS = {
    "Z": [(6.0, 1), (4.0, 4.0), (3.0, 11.8), (2, 40), (1.5, float("inf"))],
    "Y": [(3.0, 1), (2.0, float("inf"))],
    "X": [
        (6.0, 1),
        (4.0, 4.0),
        (3.0, 11.8),
        (2.5, 13.5),
        (2.0, float("inf")),
    ],  # diagram modified 0.5g inf
}

THRESHOLDS_neg1 = {  # base case  (for sled)
    "Z": [(-1.5, 3.5), (-1.1, float("inf"))],  # I.2.5
    "Y": [(-3.0, 1), (-2.0, float("inf"))],  # I.2.4
    "X": [(-1.5, float("inf"))],  # I.2.3
}

THRESHOLDS_neg2 = {  # over the shoulder restraint  (for lightning)
    "Z": [(-1.5, 3.5), (-1.1, float("inf"))],
    "Y": [(-3.0, 1), (-2.0, float("inf"))],
    "X": [(-2.0, float("inf"))],
}


# filter the Data
def butter_lowpass_filter(data, cutoff=5, fs=50, order=4):
    """
    Apply a 4-pole, single-pass Butterworth low-pass filter.

    Parameters:
    data   : array-like, raw signal
    cutoff : float, corner frequency in Hz
    fs     : float, sampling frequency in Hz
    order  : int, filter order (4-pole → order=4)

    Returns:
    filtered signal (single-pass)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = lfilter(b, a, data)  # single-pass filtering
    return y


# --- Manual Safety Check ---
def check_safety(axis: str, g_force: float, duration: float, ride_type: str) -> str:
    axis = axis.upper()
    if axis not in THRESHOLDS:
        return "Invalid axis."

    # Initialize result and allowable acceleration
    result = "⚠️ Unsafe – exceeds safety limits."
    admissible_acceleration = None

    # For positive g-force
    if g_force >= 0:
        for limit, max_time in THRESHOLDS[axis]:
            if g_force <= limit and duration <= max_time:
                return "✅ Safe"
            elif duration <= max_time:
                admissible_acceleration = (
                    limit  # Set the allowable g-force when it's unsafe
                )
                break

    # For negative g-force, based on ride type
    elif ride_type == "Over the shoulder restraint":
        for limit, max_time in THRESHOLDS_neg2[axis]:
            if g_force >= limit and duration <= max_time:
                return "✅ Safe"
            elif duration <= max_time:
                admissible_acceleration = (
                    limit  # Set the allowable g-force when it's unsafe
                )
                break

    elif ride_type == "base case":
        for limit, max_time in THRESHOLDS_neg1[axis]:
            if g_force >= limit and duration <= max_time:
                return "✅ Safe"
            elif duration <= max_time:
                admissible_acceleration = (
                    limit  # Set the allowable g-force when it's unsafe
                )
                break

    # If no safe condition is met, return the result with the allowable acceleration
    return f"{result} Allowable acceleration: {admissible_acceleration} g."


# --- Dataset Reader for IMU Format ---
def read_imu_file(uploaded_file):
    # Read raw file
    df = pd.read_csv(uploaded_file, sep=",", header=None)

    # Split time string + timestamp
    time_and_stamp = df[0].str.split(" ", expand=True)
    df.insert(0, "time_str", time_and_stamp[0])  # keep only clock time
    df = df.drop(columns=[0] + list(df.columns[5:]))  # drop old combined + unused

    # Rename columns
    df.columns = ["time_str", "acc_x", "acc_y", "acc_z"]

    # Convert time to datetime
    first_time = pd.to_datetime(df["time_str"].iloc[0], format="%H:%M:%S.%f")
    last_time = pd.to_datetime(df["time_str"].iloc[-1], format="%H:%M:%S.%f")
    total_duration = (last_time - first_time).total_seconds()

    # Time pitch and continuous timeline
    n = len(df)
    time_pitch = total_duration / (n - 1)
    df["time_sec"] = np.arange(n) * time_pitch

    # Drop original time_str
    df = df.drop(columns=["time_str"])

    return df


# --- get the admissible limit for one axis at one time ---
def admissible_limit(axis: str, g: float, ride_type: str) -> float:
    """
    Return the admissible (limit) value for the given axis at value g,
    using the correct polarity tables (positive vs negative).
    If g exceeds all admissible limits, return +inf.
    """
    pos = THRESHOLDS[axis]
    neg = (
        THRESHOLDS_neg2[axis]
        if ride_type == "Over the shoulder restraint"
        else THRESHOLDS_neg1[axis]
    )

    if g >= 0:
        # smallest positive limit >= g
        limits = sorted([L for L, _ in pos])
        for L in limits:
            if g <= L:
                return L
        return float("inf")
    else:
        # “closest to zero” negative limit <= g
        limits = sorted([L for L, _ in neg])  # e.g. [-3.0, -2.0]
        for L in reversed(limits):  # e.g. -2.0 then -3.0
            if g >= L:
                return L
        return float("inf")


# --- Combined Safety Check Function ---
def check_combined_safety_all(time_s, gx, gy, gz, ride_type: str):
    """
    Combined safety check point-by-point using the three formulas (I.1–I.3):
      (gx/adm_x)^2 + (gy/adm_y)^2 <= 1
      (gx/adm_x)^2 + (gz/adm_z)^2 <= 1
      (gz/adm_z)^2 + (gy/adm_y)^2 <= 1

    Returns:
        safe_points   : list[(t, gx, gy, gz)]
        unsafe_points : list[(t, gx, gy, gz)]
    """
    safe_pts, unsafe_pts = [], []

    for i in range(len(time_s)):
        t = float(time_s.iloc[i])
        axv = float(gx.iloc[i])
        ayv = float(gy.iloc[i])
        azv = float(gz.iloc[i])

        ax_adm = admissible_limit("X", axv, ride_type)
        ay_adm = admissible_limit("Y", ayv, ride_type)
        az_adm = admissible_limit("Z", azv, ride_type)

        # If any admissible is infinite, the point is beyond limits => unsafe
        if np.isinf(ax_adm) or np.isinf(ay_adm) or np.isinf(az_adm):
            unsafe_pts.append((t, axv, ayv, azv))
            continue

        f1 = (axv / ax_adm) ** 2 + (ayv / ay_adm) ** 2 <= 1.0
        f2 = (axv / ax_adm) ** 2 + (azv / az_adm) ** 2 <= 1.0
        f3 = (azv / az_adm) ** 2 + (ayv / ay_adm) ** 2 <= 1.0

        if f1 and f2 and f3:
            safe_pts.append((t, axv, ayv, azv))
        else:
            unsafe_pts.append((t, axv, ayv, azv))

    return safe_pts, unsafe_pts


# --- PLot Oval Shaped Combined accelerations ---


# --- calculates the admissible limits for each axis ---
def sign_limit(axis, sign, ride_type):
    """
    Return the admissible *peak* magnitude (semi-axis length) in raw g for the given axis and sign.
    sign: +1 for positive side, -1 for negative side
    """
    if sign >= 0:
        # positive side: use the *largest* positive limit in your THRESHOLDS table
        return max(L for L, _ in THRESHOLDS[axis])
    else:
        # negative side: use the *most negative* (farthest from zero) limit for the selected ride_type
        neg_tbl = (
            THRESHOLDS_neg2
            if ride_type == "Over the shoulder restraint"
            else THRESHOLDS_neg1
        )
        return abs(min(L for L, _ in neg_tbl[axis]))  # magnitude


# --- computes the ellipsoid's radius based on the semi-axes ---
def r_of_theta(theta, ax_pos, ax_neg, ay_pos, ay_neg):
    """
    Vectorized piecewise ellipsoidal radius for direction (cosθ, sinθ),
    selecting +/− semi-axes element-wise.
    """
    theta = np.asarray(theta, dtype=float)
    cx = np.cos(theta)
    sy = np.sin(theta)

    # choose semi-axes per quadrant, element-wise
    Ax = np.where(cx >= 0.0, float(ax_pos), float(ax_neg))
    Ay = np.where(sy >= 0.0, float(ay_pos), float(ay_neg))

    denom = (cx * cx) / (Ax * Ax) + (sy * sy) / (Ay * Ay)
    denom = np.maximum(denom, 1e-12)  # avoid divide-by-zero
    return 1.0 / np.sqrt(denom)


# --- checks if each point is inside the safe zone of the ellipsoid ---
def combined_safe_mask(ax_vals, ay_vals, ax_pos, ax_neg, ay_pos, ay_neg):
    """
    Safe if (ax/Ax)^2 + (ay/Ay)^2 <= 1 using sign-dependent semi-axes.
    """
    Ax = np.where(ax_vals >= 0, ax_pos, ax_neg)
    Ay = np.where(ay_vals >= 0, ay_pos, ay_neg)
    return (ax_vals / np.clip(Ax, 1e-12, None)) ** 2 + (
        ay_vals / np.clip(Ay, 1e-12, None)
    ) ** 2 <= 1.0


# --- generates the scatter plot with the ellipsoid boundary and safety check for each pair of axes ---
def pair_plot_raw(ax_vals, ay_vals, axisA, axisB, ride_type, title):
    """
    Build a Plotly scatter with the asymmetric ellipsoidal limit in RAW g.
    axisA, axisB are strings 'X'/'Y'/'Z'.
    """
    # Semi-axes for + and − sides (magnitudes in g)
    A_pos = sign_limit(axisA, +1, ride_type)
    A_neg = sign_limit(axisA, -1, ride_type)
    B_pos = sign_limit(axisB, +1, ride_type)
    B_neg = sign_limit(axisB, -1, ride_type)

    # Boundary curve (piecewise ellipse around 0)
    th = np.linspace(0, 2 * np.pi, 720)
    r = r_of_theta(th, A_pos, A_neg, B_pos, B_neg)
    bx = r * np.cos(th)
    by = r * np.sin(th)

    # Safe/unsafe split
    safe = combined_safe_mask(ax_vals, ay_vals, A_pos, A_neg, B_pos, B_neg)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=bx, y=by, mode="lines", name="Combined limit", line=dict(width=2))
    )

    fig.add_trace(
        go.Scatter(
            x=ax_vals[safe], y=ay_vals[safe], mode="markers", name="Safe", opacity=0.6
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ax_vals[~safe],
            y=ay_vals[~safe],
            mode="markers",
            name="Unsafe",
            marker=dict(color="red"),
            opacity=0.85,
        )
    )

    # 1:1 aspect to keep ellipsoid true
    fig.update_layout(
        title=title,
        xaxis_title=f"a{axisA} (g)",
        yaxis_title=f"a{axisB} (g)",
        xaxis=dict(zeroline=True),
        yaxis=dict(zeroline=True, scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h"),
    )
    return fig


# --- Dataset Safety Check ---
def check_series_safety(
    axis: str, g_series, t_series, ride_type: str, spike_series=None
):
    """
    Segment by g-threshold bins (not duration), keep segments intact even if spikes occur,
    classify segments by duration vs ISO bands (pos vs neg handled separately),
    and record single-sample spikes. If spike_series is provided, spikes are detected on it
    (e.g., raw signal) while segments use g_series (e.g., filtered).
    Returns: safe_segments (list[dict]), unsafe_segments (list[dict]), spikes (list[(t, g)])
    """
    if axis not in THRESHOLDS:
        return [], [], []

    thresholds_pos = THRESHOLDS[axis]
    thresholds_neg = (
        THRESHOLDS_neg2[axis]
        if ride_type == "Over the shoulder restraint"
        else THRESHOLDS_neg1[axis]
    )

    # Bin edges include 0 so bins never straddle the origin
    bounds = sorted({0.0, *[L for L, _ in (thresholds_pos + thresholds_neg)]})

    def get_bin(g):
        for i in range(len(bounds) - 1):
            if bounds[i] <= g < bounds[i + 1]:
                return (bounds[i], bounds[i + 1])
        return (bounds[-1], float("inf")) if g >= 0 else (-float("inf"), bounds[0])

    # For spike test, use raw if provided; otherwise use the same as segment series
    spike_g = spike_series if spike_series is not None else g_series

    # Extreme allowed values for instant "magnitude" spike test
    max_pos = max((L for L, _ in thresholds_pos), default=float("inf"))
    min_neg = min((L for L, _ in thresholds_neg), default=-float("inf"))

    eps = 1e-9

    segments = []
    spikes = []
    seg_start_idx = 0
    curr_bin = get_bin(g_series.iloc[0])

    for i in range(1, len(g_series)):
        g_seg = g_series.iloc[i]
        g_spk = spike_g.iloc[i]
        b = get_bin(g_seg)

        # Detect instantaneous spike but DO NOT cut the segment
        if g_spk >= 0:
            if g_spk > max_pos + eps:
                spikes.append((t_series.iloc[i], g_spk))
        else:
            if g_spk < min_neg - eps:
                spikes.append((t_series.iloc[i], g_spk))

        # If bin changed (crossed a limit boundary), close previous segment
        if b != curr_bin:
            g_slice = g_series.iloc[seg_start_idx:i]
            t_slice = t_series.iloc[seg_start_idx:i]
            segments.append(
                {
                    "start": t_slice.iloc[0],
                    "end": t_slice.iloc[-1],
                    "duration": t_slice.iloc[-1] - t_slice.iloc[0],
                    "g_min": g_slice.min(),
                    "g_max": g_slice.max(),
                    "bin": curr_bin,
                }
            )
            seg_start_idx = i
            curr_bin = b

    # Final segment
    g_slice = g_series.iloc[seg_start_idx:]
    t_slice = t_series.iloc[seg_start_idx:]
    segments.append(
        {
            "start": t_slice.iloc[0],
            "end": t_slice.iloc[-1],
            "duration": t_slice.iloc[-1] - t_slice.iloc[0],
            "g_min": g_slice.min(),
            "g_max": g_slice.max(),
            "bin": curr_bin,
        }
    )

    # Classify segments using correct rule per polarity
    safe_segments, unsafe_segments = [], []
    for seg in segments:
        gmin, gmax, dur = seg["g_min"], seg["g_max"], seg["duration"]
        low, high = seg["bin"]
        is_positive_bin = low >= 0 and high >= 0

        if is_positive_bin:
            # POS: need g_max ≤ limit and duration ≤ max_time
            ok = any(
                (gmax <= L + eps) and (dur <= Tmax + eps) for L, Tmax in thresholds_pos
            )
        else:
            # NEG: need g_min ≥ limit and duration ≤ max_time
            ok = any(
                (gmin >= L - eps) and (dur <= Tmax + eps) for L, Tmax in thresholds_neg
            )

        (safe_segments if ok else unsafe_segments).append(seg)

    return safe_segments, unsafe_segments, spikes


# --- Title ---
st.markdown(
    """<h1 style='color: #2196F3; font-size: 40px; text-align: center;
            font-weight: bold;'>Acceleration Effects on Passengers</h1>""",
    unsafe_allow_html=True,
)
st.markdown(
    """<h2 style='color: #2196F3;font-size: 35px; text-align: center;
            font-weight: bold;'>Medical Tolerances</h2>""",
    unsafe_allow_html=True,
)
st.markdown(
    """<h3 style='color: #2196F3; font-size: 30px; text-align: center;
            font-weight: bold;'>when USING AMUSEMENT DEVICES ⚙️</h3>""",
    unsafe_allow_html=True,
)

# --- Image and Guide ---
st.subheader("Body Coordinate System:")
AXIS_GUIDE_URL = "assets/Axis_Guide_new.png"
st.image(AXIS_GUIDE_URL, caption="3-axis X-Y-Z", use_column_width=True)

st.write(
    """
This diagram shows how the X, Y, and Z axes are oriented 
relative to the human body while seated in the amusement vehicle.  
It is used as a guide when interpreting acceleration data 
from the mounted sensors.  
"""
)

st.subheader(
    "Acceleration is defined in accordance with the following coordinate system:"
)
st.write(
    """
+a_z presses the body into the seat downwards, described as “eyes down”.

−a_z lifts the body out of the seat, described as “eyes up”.

+a_y presses the body sideward to the right, described as “eyes right”.

−a_y presses the body sideward to the left, described as “eyes left”.

+a_x presses the body into the seat backward, described as “eyes back”.

−a_x pushes the body out of the seat forward, described as “eyes front”.  
"""
)

# --- Mode Selection ---
mode = st.radio("Select Mode", ["Manual Input", "Upload Dataset"])

# --- Manual Mode ---
if mode == "Manual Input":
    ride_type = st.selectbox(
        "Ride Vehicle", ["Over the shoulder restraint", "Base case (typical restraint)"]
    )
    axis = st.selectbox("Axis", ["X", "Y", "Z"])
    g_force = st.number_input("Acceleration (g)", value=1.0)
    duration = st.number_input("Duration (seconds)", value=1.0)

    if st.button("Check Safety"):
        result = check_safety(axis, g_force, duration, ride_type)

        if "Safe" in result:
            st.success(f"Result: {result}")
        else:
            st.error(f"Result: {result}")

# --- Dataset Mode ---
else:
    ride_type = st.selectbox(
        "Ride Vehicle", ["Over the shoulder restraint", "Base case (typical restraint)"]
    )
    uploaded_file = st.file_uploader("Upload IMU CSV/TXT file", type=["csv", "txt"])

    if uploaded_file is not None:
        df = read_imu_file(uploaded_file)

        # UI: choose which axes to invert
        st.subheader("Sensor Direction")
        invert_axes = st.multiselect(
            "Invert (multiply by -1) the following axes",
            ["X", "Y", "Z"],
            default=["X", "Z"],  # make this [] if we don't want a default
        )

        # Apply inversion on raw data BEFORE filtering
        for ax in invert_axes:
            df[f"acc_{ax.lower()}"] = -df[f"acc_{ax.lower()}"]

        fs = 50  # your IMU sampling rate
        df["acc_x_filtered"] = butter_lowpass_filter(
            df["acc_x"], cutoff=5, fs=fs, order=4
        )
        df["acc_y_filtered"] = butter_lowpass_filter(
            df["acc_y"], cutoff=5, fs=fs, order=4
        )
        df["acc_z_filtered"] = butter_lowpass_filter(
            df["acc_z"], cutoff=5, fs=fs, order=4
        )

        df_preview = df[
            ["time_sec", "acc_x_filtered", "acc_y_filtered", "acc_z_filtered"]
        ]
        st.write("### Data Preview", df_preview.head(20))
        # check if you wanna see the whole data
        if st.checkbox("Show All Data After Processing", value=False):
            st.write("### Data Preview", df_preview)
        st.subheader("Note:")
        st.write(
            """
        Post-processed with a 4-pole, single pass,
        Butterworth low pass filter using a corner frequency of 5 Hz
        (Section 1.2.1, Annex I, ISO 17842-1)  
        """
        )

        results = {}
        total_unsafe_duration = 0.0

        for axis in ["X", "Y", "Z"]:
            safe_segments, unsafe_segments, spikes = check_series_safety(
                axis,
                df[f"acc_{axis.lower()}_filtered"],  # filtered → segmentation
                df["time_sec"],
                ride_type,
                spike_series=df[f"acc_{axis.lower()}"],  # raw → spike detection
            )
            results[axis] = {
                "safe": safe_segments,
                "unsafe": unsafe_segments,
                "spikes": spikes,
            }
            # Calculate total unsafe duration based on the traditional safety check
            total_unsafe_duration += sum(
                seg["duration"] for seg in unsafe_segments
            )  # sum durations

        # Combined check (do it once for all three series)
        comb_safe, comb_unsafe = check_combined_safety_all(
            df["time_sec"],
            df["acc_x_filtered"],
            df["acc_y_filtered"],
            df["acc_z_filtered"],
            ride_type,
        )
        results["combined"] = {"safe": comb_safe, "unsafe": comb_unsafe}

        # Results per axis
        st.subheader("Ride Evaluation Results")
        for axis in ["X", "Y", "Z"]:
            safe_segments = results[axis]["safe"]
            unsafe_segments = results[axis]["unsafe"]
            comb_unsafe = results["combined"]["unsafe"]  # Combined result

            # Display safe/unsafe for traditional segments
            if len(unsafe_segments) == 0:
                st.success(
                    f"{axis}-axis: ✅ Safe (all {len(safe_segments)} segments within thresholds)"
                )
            else:
                total_axis_unsafe = sum(
                    seg["duration"] for seg in unsafe_segments
                )  # ← fix
                st.error(
                    f"{axis}-axis: ⚠️ {len(unsafe_segments)} unsafe segments "
                    f"(total {total_axis_unsafe:.2f} s beyond limits)"
                )

            # Display combined unsafe check
            if axis == "Z":
                if len(comb_unsafe) == 0:
                    st.success("Combined check (I.1&I.3): ✅ Safe (no unsafe points)")
                else:
                    st.error(
                        f"Combined check (I.1&I.3): ⚠️ {len(comb_unsafe)} unsafe points"
                    )

        # Overall
        if (
            all(len(results[a]["unsafe"]) == 0 for a in ["X", "Y", "Z"])
            and len(results["combined"]["unsafe"]) == 0
        ):
            st.success("Overall: ✅ Safe")
        else:
            st.error(
                f"Overall: ⚠️ Unsafe – per-axis unsafe duration: {total_unsafe_duration:.2f} s; "
                f"combined unsafe points: {len(results['combined']['unsafe'])}"
            )

        # --- Section 1: Acceleration for Each Axis ---
        st.markdown(
            """
            <div style='width: 100%; background-color: #2196F3; padding: 10px;'>
                <h2 style='color: white; text-align: center;'>Acceleration in Each Direction</h2>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Plot for each direction acceleration
        st.subheader("Acceleration (g) over Time:")
        fig = go.Figure()

        # Plot acceleration data for each axis
        fig.add_trace(
            go.Scatter(
                x=df["time_sec"],
                y=df["acc_x_filtered"],
                mode="lines",
                name="X-axis g",
                line=dict(color="cyan"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["time_sec"],
                y=df["acc_y_filtered"],
                mode="lines",
                name="Y-axis g",
                line=dict(color="orange"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["time_sec"],
                y=df["acc_z_filtered"],
                mode="lines",
                name="Z-axis g",
                line=dict(color="green"),
            )
        )

        # Highlight unsafe segments
        for axis_name, color in zip(["X", "Y", "Z"], ["cyan", "orange", "green"]):
            unsafe_segments = results[axis_name]["unsafe"]
            for seg in unsafe_segments:
                fig.add_vrect(
                    x0=seg["start"],
                    x1=seg["end"],
                    fillcolor="red",
                    opacity=0.3,
                    line_width=0,
                    annotation_text="Unsafe",
                    annotation_position="top right",
                )

        # Plot spike points as red dots
        for axis_name, signal_color in zip(
            ["X", "Y", "Z"], ["cyan", "orange", "green"]
        ):
            spikes = results[axis_name]["spikes"]
            for t_spike, g_spike in spikes:
                fig.add_trace(
                    go.Scatter(
                        x=[t_spike],
                        y=[g_spike],
                        mode="markers",
                        marker=dict(color="red", size=8),
                        name=f"{axis_name} spike",
                    )
                )

        # Update layout for interactive plotting
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Acceleration (g)",
            template="plotly_dark",  # dark theme
            showlegend=True,
            hovermode="x unified",  # Show hover for all data points at once
            xaxis=dict(rangeslider=dict(visible=True)),  # Enable zooming
        )

        # Display the interactive plot
        st.plotly_chart(fig)

        # Table of unsafe segments
        rows = []
        for axis in ["X", "Y", "Z"]:
            for seg in results[axis]["unsafe"]:
                rows.append(
                    {
                        "Axis": axis,
                        "Start (s)": seg["start"],
                        "End (s)": seg["end"],
                        "Duration (s)": seg["duration"],
                        "g_min": seg["g_min"],
                        "g_max": seg["g_max"],
                    }
                )

        if rows:
            st.write("### Unsafe Segments")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Table of spikes
        spike_rows = []
        for axis in ["X", "Y", "Z"]:
            for t_spike, g_spike in results[axis]["spikes"]:
                spike_rows.append(
                    {
                        "Axis": axis,
                        "Time (s)": t_spike,
                        "acceleration": g_spike,
                        "Direction": "Positive" if g_spike > 0 else "Negative",
                    }
                )

        if spike_rows:
            st.write("### Detected Spikes (Above Threshold Magnitudes)")
            st.dataframe(pd.DataFrame(spike_rows), use_container_width=True)

        # Table of all segments
        if st.checkbox("Show All Segments Table (Safe + Unsafe)", value=False):
            all_segments = []
            for axis in ["X", "Y", "Z"]:
                for seg in results[axis]["safe"]:
                    all_segments.append(
                        {
                            "Axis": axis,
                            "Start (s)": seg["start"],
                            "End (s)": seg["end"],
                            "Duration (s)": seg["duration"],
                            "g_min": seg["g_min"],
                            "g_max": seg["g_max"],
                            "Status": "Safe",
                        }
                    )
                for seg in results[axis]["unsafe"]:
                    all_segments.append(
                        {
                            "Axis": axis,
                            "Start (s)": seg["start"],
                            "End (s)": seg["end"],
                            "Duration (s)": seg["duration"],
                            "g_min": seg["g_min"],
                            "g_max": seg["g_max"],
                            "Status": "Unsafe",
                        }
                    )

            if all_segments:
                st.write("### All Segments (Safe and Unsafe)")
                st.dataframe(pd.DataFrame(all_segments), use_container_width=True)

        # --- Section 2: Combined Acceleration Check ---
        st.markdown(
            """
            <div style='width: 100%; background-color: #2196F3; padding: 10px;'>
                <h2 style='color: white; text-align: center;'>Combined Acceleration Safety Check</h2>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Combined safety plot for combined acceleration check
        st.subheader("Combined Safety Check (X, Y, Z) over Time:")
        fig = go.Figure()

        # Plot all axes for combined acceleration
        fig.add_trace(
            go.Scatter(
                x=df["time_sec"],
                y=df["acc_x_filtered"],
                mode="lines",
                name="X-axis g",
                line=dict(color="cyan"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["time_sec"],
                y=df["acc_y_filtered"],
                mode="lines",
                name="Y-axis g",
                line=dict(color="orange"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["time_sec"],
                y=df["acc_z_filtered"],
                mode="lines",
                name="Z-axis g",
                line=dict(color="green"),
            )
        )

        # Highlight unsafe segments based on combined check
        # **OPTIMIZED PART:** Generate all vertical line shapes as a list of dictionaries
        vline_shapes = []
        # NOTE: We only need the t_comb (time) value from the unsafe results
        unsafe_times = [
            t_comb for (t_comb, gxv, gyv, gzv) in results["combined"]["unsafe"]
        ]

        for t_comb in unsafe_times:
            vline_shapes.append(
                dict(
                    type="line",
                    xref="x",  # Reference the x-axis
                    yref="paper",  # Reference the plot area (0 to 1)
                    x0=t_comb,
                    y0=0,
                    x1=t_comb,
                    y1=1,
                    line=dict(
                        color="red",
                        width=1,
                    ),
                )
            )

        # Apply all shapes in a single call to update_layout
        fig.update_layout(
            shapes=vline_shapes,
        )

        # Update layout for combined safety check
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Acceleration (g)",
            template="plotly_dark",  # dark theme
            showlegend=True,
            hovermode="x unified",  # Show hover for all data points at once
            xaxis=dict(rangeslider=dict(visible=True)),  # Enable zooming
        )

        # Display the combined acceleration plot
        st.plotly_chart(fig)

        # Also add combined safety status in the output tables
        combined_rows = [
            {
                "Time (s)": t,
                "acceleration X": gx,
                "acceleration Y": gy,
                "acceleration Z": gz,
                "Status": "Unsafe",
            }
            for (t, gx, gy, gz) in results["combined"]["unsafe"]
        ]
        if combined_rows:
            st.write("### Detected Unsafe Points in Combined Acceleration")
            st.dataframe(pd.DataFrame(combined_rows), use_container_width=True)

        # --- Combined pair plots (normalized) ---
        # --- Pairwise combined envelopes in RAW g (ellipsoids) ---
        gx = df["acc_x_filtered"].to_numpy()
        gy = df["acc_y_filtered"].to_numpy()
        gz = df["acc_z_filtered"].to_numpy()

        # st.markdown("""
        # <div style='width:100%; background:#2196F3; color:#fff; padding:10px; margin-top:10px;'>
        # <h3 style='margin:0; text-align:center;'>Combined Acceleration – Ellipsoidal Limits</h3>
        # </div>
        # """, unsafe_allow_html=True)
        st.subheader("Combined Acceleration – Ellipsoidal Graphs")

        col1, col2, col3 = st.columns(3)
        with col1:
            fig_xy = pair_plot_raw(gx, gy, "X", "Y", ride_type, "X–Y combined")
            st.plotly_chart(fig_xy, use_container_width=True)
        with col2:
            fig_yz = pair_plot_raw(gy, gz, "Y", "Z", ride_type, "Y–Z combined")
            st.plotly_chart(fig_yz, use_container_width=True)
        with col3:
            fig_xz = pair_plot_raw(gx, gz, "X", "Z", ride_type, "X–Z combined")
            st.plotly_chart(fig_xz, use_container_width=True)








