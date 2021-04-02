# Simulator for Idealized Polymer Conformations

Explain stuff.

import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import regex as re
from itertools import cycle
import math

bond_lengths = pd.read_csv("Bond Lengths.csv", index_col=0)

def fjc(blengths, n_mon, bpm):
    x, y, z = np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))
    for i in np.arange(n_mon*bpm):
        prev_x, prev_y, prev_z = x[i], y[i], z[i]
        blength = next(blengths)
        assert not math.isnan(blength), "No data for bond type"
        new_x = np.random.choice(np.linspace(-blength, blength, 1000))
        new_y = np.random.choice(np.linspace(-np.sqrt(blength**2-new_x**2), np.sqrt(blength**2-new_x**2), 1000))
        new_z = np.random.choice([-1, 1])*np.sqrt(blength**2-new_x**2-new_y**2)
        x, y, z = np.append(x, prev_x+new_x), np.append(y, prev_y+new_y), np.append(z, prev_z+new_z) 
    return x, y, z

def frc(blengths, n_mon, bpm):
    x, y, z = np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))
    for i in np.arange(n_mon*bpm):
        prev_x, prev_y, prev_z = x[i], y[i], z[i]
        blength = next(blengths)
        assert not math.isnan(blength), "No data for bond type"
        new_y = blength*np.cos(1.23)
        new_x = np.random.choice(np.linspace(-blength*np.sin(1.23), blength*np.sin(1.23), 1000))
        new_z = np.random.choice([-1, 1])*np.sqrt((blength**2)*((np.sin(1.23))**2) - new_x**2)
        x, y, z = np.append(x, prev_x+new_x), np.append(y, prev_y+new_y), np.append(z, prev_z+new_z) 
    return x, y, z

def simulate_conformation(model, n_mon, chain_formula):
    assert model in ["fjc", "frc", "hrc"], "Invalid conformational model"
    elements = re.findall(r"[A-Z][^A-Z]*", chain_formula)
    for element in elements:
        assert element in bond_lengths.index, "Invalid element in polymer backbone"
    bonds = []
    for i in np.arange(len(elements)-1):
        bonds.append([elements[i], elements[i+1]])
    bonds.append([elements[0], elements[-1]])
    bpm = len(bonds)
    blengths = cycle([bond_lengths.loc[bond[0], bond[1]] for bond in bonds])
    if model=="fjc":
        x_coords, y_coords, z_coords = fjc(blengths, n_mon, bpm)
    elif model=="frc":
        x_coords, y_coords, z_coords = frc(blengths, n_mon, bpm)
    elif model=="hrc":
        x_coords, y_coords, z_coords = hrc(blengths, n_mon, bpm)
    return x_coords, y_coords, z_coords

### Modeling single-chain conformational entropy and ideal free energy

def plot_conformation(model, n_mon, chain_formula):
    x_coords, y_coords, z_coords = simulate_conformation(model, n_mon, chain_formula)
    R2 = round(x_coords[-1]**2 + y_coords[-1]**2 + z_coords[-1]**2, 2)
    conformation = go.Figure(go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, marker={"size":0.1}, line={"width":2}, name="Simulated chain"))
    conformation.add_trace(go.Scatter3d(x=[0, x_coords[-1]], y=[0, y_coords[-1]], z=[0, z_coords[-1]], marker={"size":0.1}, line={"dash":"dash", "width":2}, name="End-to-end distance"))
    conformation.update_layout(scene={"xaxis":{"backgroundcolor":"white", "visible":False}, "yaxis":{"backgroundcolor":"white", "visible":False}, "zaxis":{"backgroundcolor":"white", "visible":False}}, 
                               title="{} chain simulation under {} model with degree of polymerization {} <br> Squared end-to-end distance: {} Å".format(chain_formula, model.upper(), n_mon, R2))
    conformation.show()

plot_conformation("fjc", 1000, "CC")

### Comparing conformational models

def compare_models(n_mon, chain_formula):
    fig = make_subplots(cols = 3, rows=1, 
                        specs=[[{"type":"scatter3d"}, {"type":"scatter3d"}, {"type":"scatter3d"}]], 
                        subplot_titles=["Freely Jointed Chain Model", "Freely Rotating Chain Model", "Hindered Rotation Chain Model"])

    fjc_x, fjc_y, fjc_z = simulate_conformation("fjc", n_mon, chain_formula)
    frc_x, frc_y, frc_z = simulate_conformation("frc", n_mon, chain_formula)
    hrc_x, hrc_y, hrc_z = simulate_conformation("frc", n_mon, chain_formula) #change to hrc
    
    fig.add_trace(go.Scatter3d(x=fjc_x, y=fjc_y, z=fjc_z, marker={"size":0.1}, line={"width":2}), col=1, row=1)
    fig.add_trace(go.Scatter3d(x=frc_x, y=frc_y, z=frc_z, marker={"size":0.1}, line={"width":2}), col=2, row=1)
    fig.add_trace(go.Scatter3d(x=hrc_x, y=hrc_y, z=hrc_z, marker={"size":0.1}, line={"width":2}), col=3, row=1) #change to hrc

    fw = go.FigureWidget(fig)
    fw.layout.title="{} chain simulation with degree of polymerization {}".format(chain_formula, n_mon)
    fw.layout.showlegend=False
    fw.layout.scene1={"xaxis":{"backgroundcolor":"white", "visible":False}, "yaxis":{"backgroundcolor":"white", "visible":False}, "zaxis":{"backgroundcolor":"white", "visible":False}}
    fw.layout.scene2={"xaxis":{"backgroundcolor":"white", "visible":False}, "yaxis":{"backgroundcolor":"white", "visible":False}, "zaxis":{"backgroundcolor":"white", "visible":False}}
    fw.layout.scene3={"xaxis":{"backgroundcolor":"white", "visible":False}, "yaxis":{"backgroundcolor":"white", "visible":False}, "zaxis":{"backgroundcolor":"white", "visible":False}}
    fw.show()

compare_models(1000, "CC")