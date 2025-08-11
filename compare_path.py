import pandas as pd

# Define a function to process each file
def get_average_data(file_path, metric_name):
    df = pd.read_csv(file_path)
    # Filter out non-positive values
    df_filtered = df[df['Value'] > 0]
    # Calculate the overall average for each algorithm
    avg_data = df_filtered.groupby('Algorithm')['Value'].mean().reset_index()
    avg_data.rename(columns={'Value': metric_name}, inplace=True)
    return avg_data

# Process all the datasets
avg_memory = get_average_data('memory_usage.csv', 'Average Memory Usage')
avg_time = get_average_data('execution_time.csv', 'Average Execution Time')
avg_smoothness = get_average_data('path_smoothness.csv', 'Average Path Smoothness')
avg_turns = get_average_data('turn_count.csv', 'Average Turn Count')
avg_angles = get_average_data('total_turning_angles.csv', 'Average Total Turning Angles')
avg_length = get_average_data('path_length.csv', 'Average Path Length')

# Merge the dataframes into a single summary table
summary_df = pd.merge(avg_memory, avg_time, on='Algorithm')
summary_df = pd.merge(summary_df, avg_smoothness, on='Algorithm')
summary_df = pd.merge(summary_df, avg_turns, on='Algorithm')
summary_df = pd.merge(summary_df, avg_angles, on='Algorithm')
summary_df = pd.merge(summary_df, avg_length, on='Algorithm')

# Display the summary table
print(summary_df)