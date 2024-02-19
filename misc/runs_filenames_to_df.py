import os
import pandas as pd
from glob import glob

filenames = glob('./runs/*/*/*.png')

# Define a function to parse filename
def parse_filename(filename):
    # Strip the '.png' extension and split by '_'
    parts = filename.replace('.png', '').split('_')
    
    # Extract and return the components in a dictionary
    return {
        'step': int(parts[1]),
        'btn': parts[3],
        'reward': float(parts[5]),
        'ep': float(parts[7]),
        'loc': int(parts[9]),
        'timeout': int(parts[11])
    }

# Parse all filenames
data = [parse_filename(filename) for filename in filenames]

# Create DataFrame
df = pd.DataFrame(data).sort_values(by='step')

print(df)
