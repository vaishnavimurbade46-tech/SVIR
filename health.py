import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

# Title
st.title("SVIR Epidemic Model Simulator")

# Sidebar Inputs
st.sidebar.header("Model Parameters")

N = st.sidebar.number_input("Population (N)", value=1000)
beta = st.sidebar.slider("Transmission Rate (beta)", 0.1, 1.0, 0.4)
gamma = st.sidebar.slider("Recovery Rate (gamma)", 0.05, 0.5, 0.1)
v = st.sidebar.slider("Vaccination Rate (v)", 0.0, 0.1, 0.01)

days = st.sidebar.slider("Simulation Days", 0, 300, 160)

# Initial Conditions
I0 = st.sidebar.number_input("Initial Infected", value=1)
V0 = st.sidebar.number_input("Initial Vaccinated", value=0)
R0 = st.sidebar.number_input("Initial Recovered", value=0)
S0 = N - I0 - V0 - R0

# Time grid
t = np.linspace(0, days, days)

# Model
def svir_derivative(y, t, N, beta, gamma, v):
    S, V, I, R = y
    dSdt = -beta * S * I / N - v * S
    dVdt = v * S
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dVdt, dIdt, dRdt

# Solve
y0 = (S0, V0, I0, R0)
result = odeint(svir_derivative, y0, t, args=(N, beta, gamma, v))
S, V, I, R = result.T

# Plot
st.subheader("Simulation Results")

fig, ax = plt.subplots()
ax.plot(t, S, label="Susceptible")
ax.plot(t, V, label="Vaccinated")
ax.plot(t, I, label="Infected")
ax.plot(t, R, label="Recovered")
# Create Data Table
results = odeint(model, y0, t, args=(beta, gamma))
data = pd.DataFrame({
    "Day": t.astype(int),
    "Susceptible": results[:, 0],
    "Vaccinated": results[:, 1],
    "Infected": results[:, 2],
    "Recovered": results[:, 3]
})

st.subheader("📊 Day-to-Day Simulation Data")
st.dataframe(data.style.format("{:.0f}"))

ax.set_xlabel("Days")
ax.set_ylabel("Population")
ax.legend()
ax.grid()

st.pyplot(fig)

# Metrics ("Score")
st.subheader("Key Metrics")

peak_infected = max(I)
peak_day = t[np.argmax(I)]
final_recovered = R[-1]

st.write(f"Peak Infected: {peak_infected:.0f}")
st.write(f"Peak Day: {peak_day:.0f}")
st.write(f"Total Recovered: {final_recovered:.0f}")

