import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

st.set_page_config(page_title="SVIR Epidemic Model", layout="wide")

st.title("🏥 SVIR Epidemic Model Simulator")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Model Parameters")

N = st.sidebar.number_input("Total Population (N)", value=1000)
beta = st.sidebar.slider("Transmission Rate (β)", 0.1, 1.0, 0.3)
gamma = st.sidebar.slider("Recovery Rate (γ)", 0.05, 0.5, 0.1)
vaccination_rate = st.sidebar.slider("Vaccination Rate", 0.0, 0.5, 0.05)
days = st.sidebar.slider("Simulation Days", 30, 200, 150)

# Initial Conditions
I0 = 10
R0 = 0
V0 = 0
S0 = N - I0 - R0 - V0

# ---------------- SVIR Model ----------------
def svir_model(y, t, N, beta, gamma, vaccination_rate):
    S, V, I, R = y
    dSdt = -beta * S * I / N - vaccination_rate * S
    dVdt = vaccination_rate * S
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dVdt, dIdt, dRdt

# ---------------- Generate Button ----------------
if st.button("Generate Graph & Store Data"):

    t = np.linspace(0, days, days)
    y0 = S0, V0, I0, R0

    solution = odeint(svir_model, y0, t, args=(N, beta, gamma, vaccination_rate))
    S, V, I, R = solution.T

    # ---------------- Plot Graph ----------------
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(t, S, label="Susceptible", linewidth=2)
    ax.plot(t, V, label="Vaccinated", linewidth=2)
    ax.plot(t, I, label="Infected", linewidth=2)
    ax.plot(t, R, label="Recovered", linewidth=2)

    ax.set_xlabel("Days")
    ax.set_ylabel("Population")
    ax.set_title("SVIR Epidemic Simulation")
    ax.legend()

    st.pyplot(fig)

    # ---------------- Store Data ----------------
    df = pd.DataFrame({
        "Day": t,
        "Susceptible": S,
        "Vaccinated": V,
        "Infected": I,
        "Recovered": R
    })

    # Save to CSV automatically
    df.to_csv("svir_simulation_data.csv", index=False)

    # Save in session for later viewing
    st.session_state["simulation_data"] = df

    st.success("Graph Generated & Data Stored Successfully!")

# ---------------- View Stored Data ----------------
if "simulation_data" in st.session_state:
    if st.button("View Stored Table"):
        st.dataframe(st.session_state["simulation_data"])

    st.download_button(
        "Download CSV File",
        st.session_state["simulation_data"].to_csv(index=False),
        file_name="svir_simulation_data.csv",
        mime="text/csv"
    )
