# -*- coding: utf-8 -*- 
"""
Project: IMU_Path_Visualisation
Creator: Administrator
Create time: 2020-01-14 12:19
IDE: PyCharm
Introduction: call class Vis by using input files position and euler to generate path data and gesture data and visualize
              the path data and gesture via plotly Dash
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from IMU_Path_Vis import IMU_Vis
import numpy as np


# Setup the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                        'https://codepen.io/chriddyp/pen/brPBPO.css'
                        ]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# prepare the data
pos_file = "input_file\\pos-algo0_0_hig.csv"
eul_file = "input_file\\att_euler-algo0_0.csv"
Vis = IMU_Vis.Vis(pos_file=pos_file, eul_file=eul_file)
Vis.gen_gesture_on_path(num=100)
gesture_data = Vis.get_path_gesture_data()
path_data = gesture_data[:, [0, 1, 2, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
min_value = np.min(path_data, axis=0)
max_value = np.max(path_data, axis=0)
demo_intro_md = "The scatter plot below is the result of running the t-SNE algorithm on the MNIST digits, \
nd using your own datasets, follow the instructions on the project repo to setup the local version. To learn more about how the t-SNE Explorer works, click on Learn More below."

# dictionary to store color
color = {
    "x axis": "red",
    "y axis": "green",
    "z axis": "blue",
    "path": "orange",
    "font": "black",
    "paper_bgcolor": "#ffffff",
    "background-color": "#FFFFFF"
}
name_axis = ['x axis', 'y axis', 'z axis']
trace = [go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers+lines',
    name="x axis",
    marker={'size': 1, 'color': color["x axis"]}),
    go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+lines',
        name="y axis",
        marker={'size': 1, 'color': color["y axis"]}),
    go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+lines',
        name="z axis",
        marker={'size': 1, 'color': color["z axis"]}),
    go.Scatter3d(
        x=path_data[:, 0], y=path_data[:, 1], z=path_data[:, 2],
        mode='markers+lines', customdata=[i for i in range(len(path_data))],
        name="path",
        hoverinfo="x+y+z+text",   # set point index for path data
        hovertext=["point ID:" + str(i) for i in range(len(path_data))],
        marker={'size': 1, 'color': color["path"], 'colorscale': 'Blackbody', 'opacity': 0.8, "showscale": False,
                "colorbar": {"thickness": 15, "len": 0.5, "x": 0.8, "y": 0.6, }, }),
]
figure = {"data": trace,
          "layout": go.Layout(
             height=960,
             title={"text": "IMU 3D path picture",
                    "font": {"family": "Open Sans", "size": 30, "color": color["font"]}},
             scene={"aspectmode": "manual",
                    "xaxis": {"title": {"text": "x轴(由纬度lat转换而来)",
                                        "font": {"family": "Open Sans", "size": 10, "color": color["font"]}},
                              'showticklabels': True, 'autorange': False, "color": color["font"],
                              'range': [min_value[0], max_value[0]]},
                    "yaxis": {"title": {"text": "y轴(由经度lon转换而来)",
                                        "font": {"family": "Open Sans", "size": 10, "color": color["font"]}},
                              'showticklabels': True, 'autorange': False, "color": color["font"],
                              'range': [min_value[1], max_value[1]]},
                    "zaxis": {"title": {"text": "z轴(由高度alt转换而来)",
                                       "font": {"family": "Open Sans", "size": 10, "color": color["font"]}},
                              'showticklabels': True, 'autorange': False, "color": color["font"],
                              'range': [min_value[2], max_value[2]]}
             },
             clickmode='event+select',
             paper_bgcolor="#ffffff",
             legend=dict(
                    font={"family": "Open Sans", "size": 10, "color": color["font"]},
                ),
         )}

################ Dash plot starts ################################
app.layout = html.Div(
 [
    html.Div(html.H1("IMU 3D path and gesture picture", style={"color": "#1c1c1c"}),
             style={'textAlign': "center", "width": "100%", "float": "left"}),
    html.Div(
         [
             html.Div(
                 [dcc.Dropdown(
                     id='input_number',
                     options=[{'label': i, 'value': i} for i in [300, 500, 1000]],
                     value=1000)], style={'width': '100%', "margin-top": "5px"}),
             dcc.Input(
                 id="locate_nb",
                 type="number",
                 debounce=True,
                 placeholder="定位",
                 style={"width": "100%", "margin-top": "5px"},
                 max=len(path_data),
                 min=0
             ),
             dcc.Input(
                 id="input_range",
                 type="number",
                 placeholder="跨越点",
                 style={"width": "100%", "margin-top": "5px"},
                 max=len(path_data),
                 min=0
             ),
             html.Button(
                 '前进',
                 id='Forward_Submit',
                 n_clicks_timestamp=0,
                 style={"width": "100%", "margin-top": "5px", "font-size": "15px"}
             ),
             html.Button(
                 '后退',
                 id='Rollback_Submit',
                 n_clicks_timestamp=0,
                 style={"width": "100%", "margin-top": "5px", "font-size": "15px"}
             ),
             html.Button(
                 '开始',
                 id='Play_gesture',
                 n_clicks=0,
                 n_clicks_timestamp=0,
                 style={"width": "100%", "margin-top": "5px", "font-size": "15px"}
             ),
             html.Button(
                 '暂停',
                 id='Stop_gesture',
                 n_clicks=0,
                 n_clicks_timestamp=0,
                 style={"width": "100%", "margin-top": "5px", "font-size": "15px"}
             ),
             # store the locate number value
             dcc.Store(
                 id='locate_store',
                 data=dict(value=0)
             ),
             # store the input number value
             # dcc.Store(
             #     id='input_store',
             #     data=dict(value=0)
             # ),
         ], style={"width": "10%", "float": "left", "margin-top": "150px", "margin-left": "2%"}
    ),
    html.Div([dcc.Graph(id='my_general_path', figure=figure, clickData={'points': [{'customdata': 0}]}),
              dcc.Interval(id='interval-component', interval=1 * 3000, disabled=True,  # in milliseconds
                           n_intervals=0
              )], style={"width": "64%", "float": "left", "margin-top": "150px", "margin-left": "2%"}),
    html.Div([
        html.H3("please click the path data scatter on the left side to see gesture", id="output_text"),
        dcc.Graph(id='gesture')],
             style={"width": "19%", "float": "left", "margin-left": "2%", "margin-top": "150px", "margin-right": "1%"}),
 ], style={"height": "1000px", "width": "100%", "background-color": color["background-color"]}
)



@app.callback(
    [dash.dependencies.Output("my_general_path", "figure"),
     dash.dependencies.Output('interval-component', 'disabled')],
    [dash.dependencies.Input('input_number', 'value'),
     dash.dependencies.Input('locate_nb', 'value'),
     dash.dependencies.Input('Forward_Submit', 'n_clicks_timestamp'),
     dash.dependencies.Input('Rollback_Submit', 'n_clicks_timestamp'),
     dash.dependencies.Input('Play_gesture', 'n_clicks_timestamp'),
     dash.dependencies.Input('Stop_gesture', 'n_clicks_timestamp'),
     dash.dependencies.Input('interval-component', 'n_intervals')
     ],
    [dash.dependencies.State('input_range', 'value')]
)
def plot_path_graph(input_number, locate_nb,n_clicks_fw, n_clicks_rb, n_clicks_play, n_clicks_stop, n, input_range):
    ctx = dash.callback_context
    # 获取被触发的控件
    if ctx.triggered[0]['prop_id'].split('.')[0] == 'Play_gesture' or \
            (n_clicks_play > n_clicks_fw and n_clicks_play > n_clicks_rb and n_clicks_play > n_clicks_stop):
        # 只触发play button
        print("input_number", input_number)
        if input_number is None:
            # raise PreventUpdate
            disable = True
            return figure, disable
        print("n before", n)
        n = n * input_number
        if n > len(path_data):
            # raise PreventUpdate
            disable = True
            return figure, disable
        print("Play_gesture", n)
        for j in range(3):
            figure['data'][j] = go.Scatter3d(
                        x=[path_data[n, 0], path_data[n, 3 * (j + 1)]],
                        y=[path_data[n, 1], path_data[n, 3 * (j + 1) + 1]],
                        z=[path_data[n, 2], path_data[n, 3 * (j + 1) + 2]],
                        name=name_axis[j],
                        hoverinfo="x+y+z+text",  # set point index for gesture data
                        hovertext=["point ID:" + str(n) for _ in range(3)],
                        mode="markers+lines", marker=dict(color=color[name_axis[j]], size=5)
                    )
        disable = False
    elif ctx.triggered[0]['prop_id'].split('.')[0] == 'Stop_gesture':
        # 只触发stop button
        print("stop_gesture")
        disable = True
    elif ctx.triggered[0]['prop_id'].split('.')[0] == 'locate_nb':
        if locate_nb is None:
            raise PreventUpdate
        print("locate_nb", locate_nb)
        # 只触发locate功能
        if locate_nb > len(path_data) or locate_nb < 0:
            raise PreventUpdate
        for i in range(3):
            figure['data'][i] = go.Scatter3d(
                x=[path_data[locate_nb, 0], path_data[locate_nb, 3 * (i + 1)]],
                y=[path_data[locate_nb, 1], path_data[locate_nb, 3 * (i + 1) + 1]],
                z=[path_data[locate_nb, 2], path_data[locate_nb, 3 * (i + 1) + 2]],
                name=name_axis[i],
                hoverinfo="x+y+z+text",  # set point index for gesture data
                hovertext=["point ID:" + str(locate_nb) for _ in range(3)],
                mode="markers+lines", marker=dict(color=color[name_axis[i]], size=5)
            )
        disable = True
    else:
        print("else")
        disable = True
    return figure, disable


# store the original locate number
@app.callback(
    dash.dependencies.Output('locate_store', 'data'),
    [dash.dependencies.Input('locate_nb', 'value')],
    [dash.dependencies.State('locate_store', 'data')]
)
def update_store(locate_nb, store_values):
    if locate_nb is None:
        raise PreventUpdate
    store_values['value'] = locate_nb
    return store_values


@app.callback(
     dash.dependencies.Output('locate_nb', 'value'),
    [dash.dependencies.Input('Forward_Submit', 'n_clicks_timestamp'),
     dash.dependencies.Input('Rollback_Submit', 'n_clicks_timestamp')
    ],
    [dash.dependencies.State('locate_store', 'data'),
     dash.dependencies.State('input_range', 'value')]

)
def update_locate(n_clicks_fw, n_clicks_rb, store_values, input_range):
    print("update_locate", n_clicks_fw, n_clicks_rb, store_values, input_range)
    if store_values is None or input_range is None:
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        raise PreventUpdate
    if n_clicks_fw > n_clicks_rb:
        # 执行forward功能
        store_values["value"] = store_values["value"] + input_range
    else:
        # 执行rollback 功能
        store_values["value"] = store_values["value"] - input_range
    if store_values["value"] > len(path_data) or store_values["value"] < 0:
        raise PreventUpdate
    return store_values["value"]


def create_gesture(dff):
    dff = dff.reshape(3, 3)
    trace = [go.Scatter3d(
        x=[0, dff[:, 0][i]], y=[0, dff[:, 1][i]], z=[0, dff[:, 2][i]],
        mode='markers+lines',
        name=name_axis[i],
        marker={'size': 10, 'color': color[name_axis[i]], 'colorscale': 'Blackbody', 'opacity': 0.8, "showscale": False,
                "colorbar": {"thickness": 15, "len": 0.5, "x": 0.8, "y": 0.6, },
                }) for i in range(3)]
    gesture_figure = {
        'data': trace,
        'layout': go.Layout(
                height=500,
                title={"text": "See gesture",
                       "font": {"family": "Open Sans", "size": 30, "color": color["font"]}},
                # title=f"See gesture",
                paper_bgcolor="#ffffff",
                scene={
                       "aspectmode": "cube",
                       'xaxis': {'showticklabels': True,
                                 "color": "black",
                                  'autorange': False,
                                  "title": {"font": {"family": "Open Sans", "size": 10, "color": "black"}},
                                  'range': [-1, 1],
                                 },
                       'yaxis': {'showticklabels': True,
                                 'autorange': False,
                                 "color": "black",
                                 "title": {"font": {"family": "Open Sans", "size": 10, "color": "black"}},
                                 'range': [-1, 1],
                                 },
                       'zaxis': {'showticklabels': True,
                                 'autorange': False,
                                 "color": "black",
                                 "title": {"font": {"family": "Open Sans", "size": 10, "color": "black"}},
                                 'range': [-1, 1],
                                 },
                       },
             ),
    }
    return gesture_figure, {'display': 'block'}, {'display': 'none'}



@app.callback(
    [dash.dependencies.Output("gesture", "figure"),
     dash.dependencies.Output("gesture", "style"),
     dash.dependencies.Output("output_text", "style")],
    [dash.dependencies.Input('my_general_path', 'clickData')
     ]
)
def update_gesture(clickData):
    # click event only for path trace
    if "customdata" not in clickData['points'][0].keys():
        raise PreventUpdate
    index = clickData['points'][0]['customdata']
    # 初始化不显示姿态图,返回空figure和消息
    if index == 0:
        return {'data': [],
                'layout': go.Layout(title=f"please click the path data to see gesture", )
               },  {'display': 'none'},  {'display': 'block'}
    dff = gesture_data[index, 3:12]
    return create_gesture(dff)


if __name__ == '__main__':
    app.run_server()  # debug=True
