#
# I put these functions together because I was
# copy/pasting them between notebooks.
#

from matplotlib import pyplot as plt
import plotly.graph_objects as go

###########################################
# Using plotly for interactive plot.
# It doesn't want to work if there are too
# many data points.  I find looking at a day's
# worth is helpful...
# It assumes index is timestamps.
# e.g.: plots.interactive(df['P']['2019-11-07'],'Power Readings')
###########################################
# @title Interactive Plot


def interactive(df_one_column, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_one_column.index,
                             y=df_one_column, line_color='deepskyblue'))
    fig.update_layout(title_text=title,
                      xaxis_rangeslider_visible=True)
    fig.show()

###########################################
# A simple scatter plot.
# e.g.: plots.scatter(df_2_cols,figsize=(24,8),grid=True)
###########################################


def scatter(df_2_cols, xcol=0, ycol=1, **kwargs):
    cols = len(df_2_cols.columns)
    if cols != 2:
        print('There are {} columns.  This scatter function takes in a dataframe with 2 columns'.format(cols))
        return
    if xcol == 1:
        ycol = 0
    df_2_cols.plot(kind='scatter', x=xcol, y=ycol, **kwargs)
###########################################
# Plots multiple line
# e.g.: plots.multi_line(df,figsize=(24,8),grid=True)
###########################################


def multi_line(data, cols=None, spacing=.1, **kwargs):

    from pandas import plotting

    # Get default color style from pandas - can be changed to any other color list
    if cols is None:
        cols = data.columns
    if len(cols) == 0:
        return
    colors = getattr(getattr(plotting, '_matplotlib').style,
                     '_get_standard_colors')(num_colors=len(cols))

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(
            ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
        ax_new.set_ylabel(ylabel=cols[n])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    return ax
