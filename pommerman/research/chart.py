import altair as alt
import argparse
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


colors = ['blue', 'red', 'green']

def make_success_chart(directories, bins, save_tos, max_step, ret_data=False):
    chart = None

    dataframes = {}
    for directory, save_to in zip(directories, save_tos):
        frames = []
        for f in os.listdir(directory):
            if not f.endswith('csv'):
                continue
            frames.append(pd.read_csv(os.path.join(directory, f)))

        data = pd.concat(frames)
        data['Bins'] = pd.cut(data.Step, bins=bins)

        grouped = data.groupby(data.Bins).mean()
        grouped['Std'] = data.groupby(data.Bins).std()['Value']
        grouped = grouped[grouped.Value.notnull()]
        grouped = grouped[grouped.Std.notnull()]
        newindex = np.arange(int(min(grouped['Step']))+1, int(max(grouped['Step'])))

        series = pd.Series(list(grouped.Value), index=list(grouped.Step))
        value = interp1d(series.index, series, kind='cubic')(newindex)
        series = pd.Series(list(grouped.Std), index=list(grouped.Step))
        std = interp1d(series.index, series, kind='cubic')(newindex)
        low = value - std*1.96/np.sqrt(len(frames))
        low[low < 0] = 0
        high = value + std*1.96/np.sqrt(len(frames))
        high[high > 1] = 1
        newdata = pd.DataFrame({'Epoch':newindex, 'Value':value, 'Std': std,
                                'Low':low, 'High': high})

        if ret_data:
            return grouped, newdata

        dataframes[directory] = (newdata, newindex, grouped, save_to)

    minindex = 0
    maxindex = 1000
    for directory, (_, indices, _, _) in dataframes.items():
        if indices[0] > minindex:
            minindex = indices[0]
        if indices[-1] < maxindex:
            maxindex = indices[-1]

    for directory, (newdata, _, _, save_to) in dataframes.items():
        # sliced = newdata[newdata.Step >= minindex]
        # sliced = sliced[sliced.Step <= maxindex]

        if max_step > 0:
            newdata = newdata[newdata.Epoch <= max_step]
        line = alt.Chart(newdata) \
                  .mark_line() \
                  .encode(x='Epoch', y='Value')
        conf = alt.Chart(newdata) \
                  .mark_area(opacity=0.3) \
                  .encode(
                      x='Epoch',
                      y=alt.Y('Low', axis=alt.Axis(title='Success Rate')),
                      y2='High')
        chart = conf + line
        chart.save(save_to)


def make_dual_chart(directories, names, bins, save_to, max_step, mark_bars=None, legend=True,
                    width=0, height=0, ret_data=False):
    chart = None

    dataframes = []
    for directory, name in zip(directories, names):
        frames = []
        for f in os.listdir(directory):
            if not f.endswith('csv'):
                continue
            frames.append(pd.read_csv(os.path.join(directory, f)))

        data = pd.concat(frames)
        data['Bins'] = pd.cut(data.Step, bins=bins)

        grouped = data.groupby(data.Bins).mean()
        grouped['Std'] = data.groupby(data.Bins).std()['Value']
        grouped = grouped[grouped.Value.notnull()]
        grouped = grouped[grouped.Std.notnull()]
        newindex = np.arange(int(min(grouped['Step']))+1, int(max(grouped['Step'])))

        series = pd.Series(list(grouped.Value), index=list(grouped.Step))
        value = interp1d(series.index, series, kind='cubic')(newindex)
        value[value < 0] = 0
        value[value > 1] = 1
        series = pd.Series(list(grouped.Std), index=list(grouped.Step))
        std = interp1d(series.index, series, kind='cubic')(newindex)
        low = value - std*1.96/np.sqrt(len(frames))
        low[low < 0] = 0
        high = value + std*1.96/np.sqrt(len(frames))
        high[high > 1] = 1
        newdata = pd.DataFrame({'Epoch':newindex, 'Value':value, 'Std': std,
                                'Low':low, 'High': high})
        newdata['Model'] = name

        if ret_data:
            return grouped, newdata

        if max_step > 0:
            newdata = newdata[newdata.Epoch <= max_step]
        dataframes.append((directory, newdata, name))

    bigdataframe = pd.concat([nd for _, nd, _ in dataframes])
    if legend:
        color = alt.Color('Model')
    else:
        color = alt.Color('Model', legend=None)
    line = alt.Chart(bigdataframe) \
              .mark_line() \
              .encode(x='Epoch',
                      y=alt.Y('Value', scale=alt.Scale(domain=[0., 1.])),
                      color=color,
              )
    if mark_bars:
        bars = pd.DataFrame([
            {"bar": int(bar), "color": num} for num, bar in enumerate(mark_bars)
        ])
        bar = alt.Chart(bars).mark_rule().encode(
            x=alt.X('bar', axis=None),# alt.Axis(title=None)
            # size=alt.value(32),
            color=alt.Color('color', legend=None,  
                            scale=alt.Scale(domain=['0', '1'],
                                            range=['red', 'green']))
        )
    color = alt.Color('Model', legend=None) # if not legend else alt.Color('Model')
    conf = alt.Chart(bigdataframe) \
              .mark_area(opacity=0.3) \
              .encode(
                  x='Epoch',
                  y=alt.Y('Low', scale=alt.Scale(domain=[0., 1.]),
                          axis=alt.Axis(title='Success Rate')),
                  y2='High',
                  # y2=alt.Y('High', scale=alt.Scale(domain=[0., 1.])),
                  color=color)
    chart = conf + line
    if mark_bars:
        chart += bar

    if width and height:
        chart = chart.properties(width=width, height=height)
    # if legend:
    #     chart.configure_legend(labelFontSize=3, titleFontSize=3)
    chart.save(save_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='chartsandshit')
    parser.add_argument('--func', type=int, default=0,
                        help='0 for make_success_chart, 1 for make_dual_chart')
    parser.add_argument('--directories', type=str, default='')
    parser.add_argument('--bins', type=int, default=100)
    parser.add_argument('--names', type=str, default='')
    parser.add_argument('--save-tos', type=str, default='')
    parser.add_argument('--max-step', type=int, default=0)
    parser.add_argument('--width', type=int, default=0)
    parser.add_argument('--height', type=int, default=0)
    parser.add_argument('--mark-bar', type=str, default='')
    parser.add_argument('--legend', action='store_true',
                        default=False) 
    parser.add_argument('--ret-data', action='store_true',
                        default=False)
    args = parser.parse_args()
    directories = args.directories.split(',')
    save_tos = args.save_tos.split(',')
    names = args.names.split(',')
    bars = args.mark_bar.split(',') if args.mark_bar else None
    print(bars)
    if args.func == 0:
        make_success_chart(directories, args.bins, save_tos,
                           args.max_step, args.ret_data)
    else:
        make_dual_chart(directories, names, args.bins, save_tos[0],
                        args.max_step, bars, args.legend, args.width,
                        args.height, args.ret_data)
                        
