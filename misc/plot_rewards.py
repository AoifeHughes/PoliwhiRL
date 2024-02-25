# -*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt
import sys

# Check if the command line argument is provided
if len(sys.argv) < 2:
    print("Usage: python script.py training_log.json")
    sys.exit(1)

# The first command line argument is the file name
file_name = sys.argv[1]

# Read the JSON file
with open(file_name, "r") as file:
    data = json.load(file)

# Extract epsilon_values
epsilon_values = data["rewards"]

# Plot the epsilon_values
plt.plot(epsilon_values)
plt.title("Epsilon Values Over Time")
plt.xlabel("Time Step")
plt.ylabel("Epsilon Value")
plt.show()
