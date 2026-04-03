import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

st.title("SVIR Epidemic Model Simulator")

# Sidebar Inputs
st.sidebar.header("Model Parameters")

N = st.sidebar.number_input("Total Population", value=1000)
beta = st.sidebar.slider("Transmission Rate (beta)", 0.1, 1.0, 0.5)
gamma = st.sidebar.slider("Recovery Rate (gamma)", 0.1, 1.0, 0.3)
vaccination_rate = st.sidebar.slider("Vaccination Rate", 0.0, 0.5, 0.05)
days = st.sidebar.slider("Simulation Days", 30, 200, 150)

# Initial Conditions
S0 = N - 1
V0 = 0
I0 = 1
R0 = 0

# SVIR Model
def svir_model(y, t, N, beta, gamma, vaccination_rate):
    S, V, I, R = y
    
    dSdt = -beta * S * I / N - vaccination_rate * S
    dVdt = vaccination_rate * S
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    
    return dSdt, dVdt, dIdt, dRdt

t = np.linspace(0, days, days)
y0 = S0, V0, I0, R0

solution = odeint(svir_model, y0, t, args=(N, beta, gamma, vaccination_rate))
S, V, I, R = solution.T

# ---- GRAPH FIRST ----
st.subheader("Simulation Graph")

fig, ax = plt.subplots()
ax.plot(t, S, label="Susceptible")
ax.plot(t, V, label="Vaccinated")
ax.plot(t, I, label="Infected")
ax.plot(t, R, label="Recovered")

ax.set_xlabel("Days")
ax.set_ylabel("Population")
ax.legend()

st.pyplot(fig)

# ---- SAVE DATA AUTOMATICALLY ----
df = pd.DataFrame({
    "Day": t,
    "Susceptible": S,
    "Vaccinated": V,
    "Infected": I,
    "Recovered": R
})

if not os.path.exists("saved_data"):
    os.makedirs("saved_data")

file_path = "saved_data/svir_simulation.csv"
df.to_csv(file_path, index=False)

# ---- BUTTON TO VIEW STORED TABLE ----
st.markdown("---")

if st.button("Show Stored Data Table"):
    stored_df = pd.read_csv(file_path)
    st.dataframe(stored_df)
