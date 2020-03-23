 
__author__ = 'Elnaz Azmi'
__email__ = 'elnaz.azmi@kit.edu'
__status__ = 'Development'

import numpy as np
import pandas as pd
from operator import itemgetter
import itertools
import matplotlib.pyplot as plt
from bokeh.palettes import d3 as palette
from bokeh.io import export_svgs
from bokeh.models import LinearAxis, DatetimeAxis, Range1d
from bokeh.models import Legend, BasicTickFormatter
from bokeh.plotting import figure

# outlet visualization
def outlet_vis(df, forcing, out_name, y_range=(0.008, 0.023), extra_y_range=0.14):
    df = df / 5  # scale m³/5min to m³/min
    date = df.index.values
    color = itertools.cycle(palette['Category10'][10])

    p = figure(plot_width=960, plot_height=600, x_range=(date[0], date[-1]), y_range=y_range,
               x_axis_type='datetime', x_axis_label='Date', y_axis_label='Catchment Outlet Discharge [m³/min]')
    legend_it = []

    for i in range(0, df.shape[1]):
        if df.columns.values[i] == 'Original':
            pl = p.line(df.index.values, df.iloc[:, i], line_color='#666666', line_width=2)
        else:
            c = next(color)
            pl = p.line(df.index.values, df.iloc[:, i], line_color=c, line_width=2)
        legend_it.append((df.columns.values[i], [pl]))

    p.xaxis.major_label_text_font_size = "16pt"
    p.xaxis.axis_label_text_font_size = "16pt"
    p.yaxis.formatter = BasicTickFormatter(use_scientific=True, power_limit_low=1, power_limit_high=2,
                                           precision=1)

    df_forcing = pd.DataFrame(forcing[0: date.shape[0]], columns=['forcing'])
    df_forcing = df_forcing.set_index(date)
    df_forcing = df_forcing / 5  # scale mm/5min to mm/min

    # Setting the second axis range name and range
    p.extra_y_ranges = {"foo": Range1d(start=extra_y_range, end=0.0)}
    p.extra_x_ranges = {"faa": Range1d(date[0], date[-1])}

    # Adding the second axis to the plot
    p.add_layout(LinearAxis(y_range_name="foo", axis_label="Rainfall [mm/min]"), 'right')
    p.add_layout(DatetimeAxis(x_range_name="faa", axis_label="", major_tick_line_color=None,
                              major_label_text_font_size='0pt'), 'above')

    pl = p.vbar(x=df_forcing.index.values, top=df_forcing['forcing'], x_range_name='faa', y_range_name='foo',
                line_color='#00BFFF', fill_color='#00BFFF', width=240000)
    legend_it.append(('Rainfall', [pl]))
    legend_it = sorted(legend_it, key=itemgetter(0))

    # turn off y-axis minor ticks
    p.yaxis.minor_tick_line_color = None

    legend = Legend(items=legend_it, location="center")
    p.add_layout(legend, 'above')
    p.legend.orientation = "horizontal"

    p.legend.label_text_font_size = "16pt"
    p.yaxis.major_label_text_font_size = "16pt"
    p.yaxis.axis_label_text_font_size = "16pt"

    p.xgrid.visible = False
    p.ygrid.visible = False

    p.output_backend = "svg"
    export_svgs(p, filename=out_name)

# rmse-speedup visualization
def rmse_speedup(df_rmse, df_runtime_s, out_name):
    fig, ax = plt.subplots(figsize=(5, 3))

    df = pd.concat([df_rmse, df_runtime_s], axis=1, sort=False)
    df.plot.scatter(x='Speedup', y='RMSE', ax=ax, s=80)

    for i, txt in enumerate(df.index.values):
        ax.annotate(txt, (df.iloc[i, 1], df.iloc[i, 0]), size=12)

    ax.set_ylim(-0.0008, 0.008)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(12)
    axis_font = {'size': '12'}
    plt.xlabel('Speedup', labelpad=6, **axis_font)
    plt.ylabel('RMSE', labelpad=8, **axis_font)

    fig.savefig(out_name, bbox_inches='tight')
