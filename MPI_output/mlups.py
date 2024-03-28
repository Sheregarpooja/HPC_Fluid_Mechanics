import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data extracted from the file
data = {
    "lattices": [200, 200, 200, 200, 200, 200, 400, 400, 400, 400, 400, 400, 400, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 800, 800, 800, 800, 800, 800, 800, 1000, 1000, 1000, 1000, 1000, 1000],
    "decomp": [2, 4, 5, 8, 10, 20, 2, 4, 5, 8, 10, 16, 20, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 2, 4, 5, 8, 10, 16, 20, 2, 4, 5, 8, 10, 20],
    "MLUPS": [11111111, 28571428, 28571428, 33333333, 40000000, 17391304, 11188811, 35555555, 47058823, 84210526, 114285714, 133333333, 106666666, 9160305, 18652849, 31578947, 38709677, 51428571, 94736842, 150000000, 189473684, 240000000, 211764705, 8951048, 29357798, 34224598, 82051282, 145454545, 320000000, 320000000, 8156606, 24330900, 30864197, 70422535, 129870129, 454545454]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Calculating the number of processors as the square of decomp
df['processors'] = df['decomp']**2

# Creating a colormap to differentiate the lattice sizes
colors = plt.cm.viridis(np.linspace(0, 1, len(lattice_sizes)))

fig, ax = plt.subplots(figsize=(10, 6))

for i, lattice in enumerate(lattice_sizes):
    subset = df[df['lattices'] == lattice]
    ax.plot(subset['processors'], subset['MLUPS'], marker='o', linestyle='-', color=colors[i], label=f'Lattice Size {lattice}')

# Setting x-axis to be logarithmic
ax.set_xscale('log')
ax.set_yscale('log')

# Customizing x-axis indices to represent processor counts more clearly
processor_counts = df['processors'].unique()
processor_counts.sort()
ax.set_xticks(processor_counts)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

# Setting labels and title
ax.set_xlabel('Number of Processors')
ax.set_ylabel('MLUPS (Log Scale)')
ax.set_title('MLUPS vs. Processors for Different Lattice Sizes')
ax.legend()
plt.grid(True, which="both", ls="--")

# Rotating x-axis labels for better visibility
plt.xticks(rotation=45)
plt.show()


