import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# ----------  Cordic core -----------------------------------------------
def cordic_hist(angle_deg: float, n: int):
    """
    Return x_hist, y_hist for CORDIC rotation-mode with n iterations.
    angle_deg may be between -90 and +90.
    """
    tan_tbl = 2.0**-(np.arange(n))  # tan φ_i  = 2^-i
    phi_tbl = np.degrees(np.arctan(tan_tbl))  # φ_i in degrees
    K = np.prod(np.cos(np.radians(phi_tbl)))  # scale factor

    x, y, theta = 1.0, 0.0, angle_deg
    x_hist = [x]
    y_hist = [y]

    for i in range(n):
        d = np.sign(theta)
        x, y = x - d * y * tan_tbl[i], y + d * x * tan_tbl[i]
        theta -= d * phi_tbl[i]
        x_hist.append(x)
        y_hist.append(y)

    # scale final values
    x_hist[-1] *= K
    y_hist[-1] *= K
    return np.array(x_hist), np.array(y_hist), phi_tbl, tan_tbl, K


# ----------  Session‑state helpers -------------------------------------
if "cur_step" not in st.session_state:
    st.session_state.cur_step = 0
if "params" not in st.session_state:
    st.session_state.params = {"angle": 30.0, "n": 6}


def reset_hist():
    st.session_state.cur_step = 0


def next_step():
    if st.session_state.cur_step < st.session_state.params["n"]:
        st.session_state.cur_step += 1


def back_step():
    if st.session_state.cur_step > 0:
        st.session_state.cur_step -= 1


# ----------  Sidebar inputs --------------------------------------------
st.sidebar.header("CORDIC controls")

angle_deg = st.sidebar.number_input("Target angle θ (degrees)",
                                    min_value=-89.99,
                                    max_value=89.99,
                                    value=st.session_state.params["angle"],
                                    step=1.0)
n_steps = st.sidebar.slider("Iterations n",
                            min_value=1,
                            max_value=30,
                            value=st.session_state.params["n"])

# Apply changes: if inputs changed -> recompute history
if angle_deg != st.session_state.params[
        "angle"] or n_steps != st.session_state.params["n"]:
    st.session_state.params = {"angle": angle_deg, "n": n_steps}
    reset_hist()

# ----------  Navigation buttons ----------------------------------------
st.sidebar.button("◀ Back", on_click=back_step)
st.sidebar.button("Next ▶", on_click=next_step)
st.sidebar.button("Reset", on_click=reset_hist)

cur = st.session_state.cur_step
θ = st.session_state.params["angle"]
n = st.session_state.params["n"]

# ----------  Run CORDIC & gather data ----------------------------------
x_hist, y_hist, phi_tbl, tan_tbl, K = cordic_hist(θ, n)

# ----------  Plot -------------------------------------------------------
fig, ax = plt.subplots(figsize=(4, 4))
ax.set_aspect("equal")
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.grid(True, alpha=0.2)

# unit circle
circle = plt.Circle((0, 0), 1, fill=False, linestyle=":")
ax.add_artist(circle)

# target vector
ax.plot([0, math.cos(math.radians(θ))], [0, math.sin(math.radians(θ))],
        "b",
        lw=2,
        label="target")

# current guess vector
ax.plot([0, x_hist[cur]], [0, y_hist[cur]], "r", lw=2, label=f"step {cur}")

ax.legend()
st.pyplot(fig)

# ----------  Right‑hand info panel -------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.metric("Current step", cur)
    st.metric("φᵢ (deg)", 0.0 if cur == n else round(phi_tbl[cur], 6))
    st.metric("tan φᵢ", 0.0 if cur == n else round(tan_tbl[cur], 6))
with col2:
    # Direction lamp
    if cur == n:
        st.markdown("**Direction:** 🔵 (done)")
    else:
        direction = np.sign(θ - math.degrees(math.atan2(y_hist[cur], x_hist[cur])))
        st.markdown(f"**Direction:** {'🟢 CCW' if direction>=0 else '🔴 CW'}")
    st.metric("K (scale factor)", round(K, 6))

# ----------  NEW final‑value metrics -----------------------------------
if cur == n:
    cordic_sin = y_hist[-1]
    python_sin = math.sin(math.radians(θ))
    err        = abs(cordic_sin - python_sin)

    st.metric("cos θ (CORDIC)", round(x_hist[-1], 10))
    st.metric("sin θ (CORDIC)", round(cordic_sin, 10))
    st.metric("sin θ (Python)", round(python_sin, 10))
    st.metric("abs error",      f"{err:.2e}")


st.caption("Green = counter-clockwise, Red = clockwise, Blue = finished")
