import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Load the dataset
df = pd.read_csv('../../Data/SplitData/training_set.csv')

# Calculate the actual spread and total points
df['actual_spread'] = df['home_score'] - df['away_score']
df['total_points'] = df['home_score'] + df['away_score']

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=('Given Spread vs Actual Spread', 'Given Total vs Total Points'))

# Spread plot
fig.add_trace(
    go.Scatter(x=df['given_spread'], y=df['actual_spread'], mode='markers', name='Spread'),
    row=1, col=1
)

# Total points plot
fig.add_trace(
    go.Scatter(x=df['given_total'], y=df['total_points'], mode='markers', name='Total Points'),
    row=1, col=2
)

# Update xaxis properties
fig.update_xaxes(title_text="Given Spread", row=1, col=1)
fig.update_xaxes(title_text="Given Total", row=1, col=2)

# Update yaxis properties
fig.update_yaxes(title_text="Actual Spread", row=1, col=1)
fig.update_yaxes(title_text="Actual Total Points", row=1, col=2)

# Update titles and layout
fig.update_layout(title_text="Expected vs Actual Performance in NFL Betting", showlegend=False)

# Show the plot
fig.show()
