import streamlit as st
import numpy as np
from ACO import ACO

st.title("Ant Colony Optimization - Path Finder")

rows = st.number_input("Enter number of rows", min_value=2, max_value=20, value=5)
cols = st.number_input("Enter number of columns", min_value=2, max_value=20, value=5)

st.write("Enter the grid values:")

grid_vals = []
for i in range(int(rows)):
    row = st.text_input(f"Row {i+1} (space-separated):", value=" ".join(["0"] * int(cols)))
    grid_vals.append(list(map(int, row.strip().split())))

grid = np.array(grid_vals)

source_val = st.number_input("Enter Source Value (e.g., 1)", value=1)
destination_val = st.number_input("Enter Destination Value (e.g., 9)", value=9)

if st.button("Run ACO"):
    aco = ACO(grid)
    path, cost = aco.run(start_val=source_val, end_val=destination_val)

    if path:
        # Convert all steps to native Python ints for clean display
        cleaned_path = [tuple(map(int, step)) for step in path]
        st.success(f"Best Path Cost: {cost}")
        st.write(" → ".join([str(p) for p in cleaned_path]))
    else:
        st.error("No valid path found!")
