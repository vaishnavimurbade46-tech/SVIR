import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.title("🏥 Hospital Vaccination Data Storage System")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Model Parameters")

N = st.sidebar.number_input("Total Population (N)", value=1000)
beta = st.sidebar.slider("Transmission Rate (β)", 0.1, 1.0, 0.4)
gamma = st.sidebar.slider("Recovery Rate (γ)", 0.1, 1.0, 0.2)
vaccination_rate = st.sidebar.slider("Vaccination Rate", 0.0, 0.1, 0.02)
days = st.sidebar.slider("Simulation Days", 10, 200, 60)

I0 = st.sidebar.number_input("Initial Infected", value=10)
R0 = 0
S0 = N - I0 - R0

# -----------------------------
# Simulation Function
# -----------------------------
def run_simulation(v_rate):
    S = [S0]
    I = [I0]
    R = [R0]

    for t in range(days):
        new_infected = beta * S[-1] * I[-1] / N
        new_recovered = gamma * I[-1]
        new_vaccinated = v_rate * S[-1]

        S.append(S[-1] - new_infected - new_vaccinated)
        I.append(I[-1] + new_infected - new_recovered)
        R.append(R[-1] + new_recovered)

    return S, I, R

# Run simulation when button clicked
if st.button("Generate Graph & Store Data"):

    S, I, R = run_simulation(vaccination_rate)

    # Create Data Table
    df = pd.DataFrame({
        "Day": range(days+1),
        "Susceptible": S,
        "Infected": I,
        "Recovered": R
    })

    # Store in session state (so it remains saved)
    st.session_state["stored_data"] = df

    # Show Graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Day"], y=df["Infected"], mode='lines', name='Infected'))
    fig.update_layout(title="Infected Population Over Time")
    st.plotly_chart(fig)

    st.success("Data Stored Successfully!")

# -----------------------------
# Show Stored Data Anytime
# -----------------------------
if "stored_data" in st.session_state:

    st.subheader("📊 Stored Simulation Data")
    st.dataframe(st.session_state["stored_data"])

    # Download Button
    csv = st.session_state["stored_data"].to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇ Download Data as CSV",
        data=csv,
        file_name="hospital_simulation_data.csv",
        mime="text/csv",
    )
