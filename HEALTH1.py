import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

st.set_page_config(page_title="SVIR Epidemic Model", layout="wide")

st.title("🏥 SVIR Epidemic Model Simulator")

# ---------------- Sidebar ----------------
st.sidebar.header("Model Parameters")

N = st.sidebar.number_input("Population (N)", min_value=1, value=1000)

beta = st.sidebar.slider("Transmission Rate (beta)", 0.1, 1.0, 0.4)
gamma = st.sidebar.slider("Recovery Rate (gamma)", 0.01, 0.5, 0.1)
nu = st.sidebar.slider("Vaccination Rate (ν)", 0.0, 0.5, 0.05)

days = st.sidebar.slider("Simulation Days", 30, 300, 160)

st.sidebar.subheader("Initial Conditions")

I0 = st.sidebar.number_input("Initial Infected", min_value=0, value=1)
V0 = st.sidebar.number_input("Initial Vaccinated", min_value=0, value=0)
R0 = st.sidebar.number_input("Initial Recovered", min_value=0, value=0)

S0 = N - I0 - V0 - R0

# ---------------- SVIR Model ----------------
def svir_model(y, t, N, beta, gamma, nu):
    S, V, I, R = y
    dSdt = -beta * S * I / N - nu * S
    dVdt = nu * S
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dVdt, dIdt, dRdt

# ---------------- Simulation ----------------
t = np.linspace(0, days, days)
y0 = S0, V0, I0, R0

solution = odeint(svir_model, y0, t, args=(N, beta, gamma, nu))
S, V, I, R = solution.T

# ---------------- Plot Graph ----------------
fig, ax = plt.subplots(figsize=(10,6))

# Strong distinct colors
ax.plot(t, S, color="blue", linewidth=3, label="Susceptible")
ax.plot(t, V, color="orange", linewidth=3, label="Vaccinated")
ax.plot(t, I, color="red", linewidth=3, label="Infected")
ax.plot(t, R, color="green", linewidth=3, label="Recovered")

ax.set_xlabel("Days", fontsize=12)
ax.set_ylabel("Population", fontsize=12)
ax.set_title("SVIR Epidemic Simulation", fontsize=15)
ax.legend(fontsize=11)
ax.grid(True)

st.pyplot(fig)

# ---------------- Peak Info ----------------
peak_infected = max(I)
peak_day = t[np.argmax(I)]

st.markdown("### 📊 Peak Infection Details")
st.write(f"Peak Infected Population: {int(peak_infected)}")
st.write(f"Peak Day: {int(peak_day)}")
