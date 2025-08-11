import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_algorithm_comparison(csv_file_path):
    """
    Reads path length data from a CSV, calculates the average path length
    for each algorithm on each map, and plots the results as a grouped
    bar chart for clear comparison.

    Args:
        csv_file_path (str): The path to the input CSV file.
    """
    try:
        # Load the dataset from the specified CSV file.
        df = pd.read_csv(csv_file_path)

        # Group the data by 'Map' and 'Algorithm', then calculate the mean
        # of the 'Value' (path length) for each group.
        # The .unstack() method pivots the 'Algorithm' level of the index
        # to become columns, which is the format needed for a grouped bar chart.
        avg_path_length = df.groupby(['Map', 'Algorithm'])['Value'].mean().unstack()

        # --- Plotting the data ---
        # Create a figure and a set of subplots.
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot the data as a bar chart.
        avg_path_length.plot(kind='bar', ax=ax, width=0.8)

        # --- Customizing the plot for clarity and aesthetics ---
        # Add a title to the plot.
        ax.set_title('Average Path Length Comparison by Algorithm and Map', fontsize=18, pad=20)
        # Add a descriptive label for the y-axis.
        ax.set_ylabel('Average Path Length', fontsize=14)
        # Add a label for the x-axis.
        ax.set_xlabel('Map ID', fontsize=14)

        # Set the x-tick labels to be the map numbers.
        ax.set_xticklabels(avg_path_length.index, rotation=0, fontsize=12)

        # Customize the legend.
        ax.legend(title='Algorithm', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

        # Add a grid for the y-axis for easier value reading.
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
        ax.set_axisbelow(True) # Ensure grid is behind bars

        # Add data labels on top of each bar for precise values.
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', label_type='edge', fontsize=8, padding=3)

        # Adjust layout to prevent labels from being cut off.
        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust rect to make space for legend

        # Display the plot.
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Main execution ---
if __name__ == '__main__':
    # Define the path to your CSV file.
    csv_file = 'path_length.csv'
    plot_algorithm_comparison(csv_file)
