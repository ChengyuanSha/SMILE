import dash
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from data_processing._processing_funcs import ResultProcessing

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)

app.title = 'LGP'

# setting up get result class
result = ResultProcessing("dataset/RuiJin_Processed.csv", "dataset/lgp_filtered.pkl")
result.load_models()
X, y, names = result.X, result.y, result.names
index_list = [i for i in range(len(names))]
available_indicators = list(zip(index_list, names))

server = app.server

# style for the text box
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

# need to suppress bc using dynamic tabs
app.config['suppress_callback_exceptions'] = True

app.layout = html.Div(
    [
        # --- headline ---
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("gene.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Linear Genetic Programming",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Result Visualization", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("Github", id="learn-more-button"),
                            href="https://github.com/ChengyuanSha/linear_genetic_programming",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),

        dcc.Tabs(id="tabs", value='tab-1', children=[
            dcc.Tab(label='Overview', value='tab-1'),
            dcc.Tab(label='Length Specific Graph', value='tab-2'),
        ]),

        html.Div(id='tabs-content')

    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"}
)


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return overview()
    elif tab == 'tab-2':
        return html.Div([
            html.H3('TODO')
        ])


def overview():
    return html.Div([

        get_prog_len_slider(),
        # 1 row 2 graphs in website
        html.Div([
            html.Div([
                dcc.Graph(
                    id='sliderfilter-occurrences-scatter',
                )
            ],
                id="left-column",
                className="pretty_container six columns",
            ),

            html.Div([
                dcc.Graph(
                    id='sliderfilter-accuracy-scatter',
                )
            ],
                id="right-column",
                className="pretty_container six columns",
            ),
        ],
            className="row flex-display",
        ),

        # 2 row Two Feature Co-occurrence in website
        html.Div([
            html.Div([
                html.Div(
                    html.H6('Two Feature Co-occurrence Analysis')
                ),

                html.Div(
                    dcc.Graph(
                        id='co-occurrence-graph'
                    )
                )

            ],
                className="pretty_container eleven columns"
            )

        ],
            className="row flex-display",
            style={'align-items': 'center', 'justify-content': 'center'},
        ),

        # 3 row selectors in website
        html.Div([
            html.Div([
                html.Div(
                    html.H6('x axis' + ' / ' + 'y axis')
                ),
                html.Div([
                    dcc.Dropdown(
                        id='crossfilter-xaxis-column',
                        options=[{'label': str(i) + ': ' + n, 'value': i} for i, n in available_indicators],
                        value='0'
                    ),
                    dcc.RadioItems(
                        id='crossfilter-xaxis-type',
                        options=[{'label': i, 'value': i} for i in ['Linear']],
                        value='Linear',
                        labelStyle={'display': 'inline-block'}
                    )
                ], style={'width': '49%', 'display': 'inline-block'},

                ),

                html.Div([
                    dcc.Dropdown(
                        id='crossfilter-yaxis-column',
                        options=[{'label': str(i) + ': ' + n, 'value': i} for i, n in available_indicators],
                        value='1'
                    ),
                    dcc.RadioItems(
                        id='crossfilter-yaxis-type',
                        options=[{'label': i, 'value': i} for i in ['Linear']],
                        value='Linear',
                        labelStyle={'display': 'inline-block'}
                    )
                ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'},
                )
            ],
                className="pretty_container twelve columns"
            ),
        ],
            className="row flex-display"
        ),

        # 4 row Two Feature Comparision in the website
        html.Div([
            # 2 feature scatter plot, update based on x, y filter, see the callback
            html.Div([
                dcc.Graph(
                    id='crossfilter-indicator-scatter',
                    # hoverData={'points': [{'customdata': 'Japan'}]}
                )
            ],
                className="pretty_container six columns",
            ),
            # display program info on the right
            html.Div([
                dcc.Markdown("""
                                **Click Data**  
                        
                                Click on points in the graph.
                            """),
                html.Pre(id='click-data', style=styles['pre']),
            ],
                className="pretty_container six columns",
            ),

        ],
            className="row flex-display",
        ),

    ])


def get_prog_len_slider():
    result.calculate_featureList_and_calcvariableList()
    length_list = sorted(list(set([len(i) for i in result.feature_list])))
    return html.Div([

        html.Div([
            html.H6('Choose Program Length'),

            dcc.Slider(
                id='proglenfilter-slider',
                min=length_list[0],
                max=length_list[-1],
                value=len(length_list),
                marks={str(each_len): str(each_len) for each_len in length_list},
                step=None
            )

        ],
            className="pretty_container ten columns",
        ),
    ],
        style={'align-items': 'center', 'justify-content': 'center'},
        className="row flex-display",
    )


@app.callback(
    dash.dependencies.Output('sliderfilter-accuracy-scatter', 'figure'),
    [dash.dependencies.Input('proglenfilter-slider', 'value')])
def update_accuracy_graph(pro_len):
    prog_index, acc_scores = result.get_accuracy_given_length(pro_len)
    prog_index = ['m' + str(i) for i in prog_index]
    return {
        'data': [
            {'x': prog_index, 'y': acc_scores, 'type': 'bar'}
        ],
        'layout': {
            'title': 'Accuracy of ' + str(pro_len) + ' Feature Models',
            'xaxis': {'title': 'model index'},
            'yaxis': {'title': 'accuracy'},
        },
    }


@app.callback(
    dash.dependencies.Output('sliderfilter-occurrences-scatter', 'figure'),
    [dash.dependencies.Input('proglenfilter-slider', 'value')])
def update_occurrence_graph(pro_len):
    features, num_of_occurrences = result.get_occurrence_from_feature_list_given_length(pro_len)
    features = ['f' + str(i) for i in features]
    return {
        'data': [
            {'x': features, 'y': num_of_occurrences, 'type': 'bar'}
        ],
        'layout': {
            'title': 'Occurrences of Features of ' + str(pro_len) + ' Feature Models',
            'xaxis': {'title': 'Program feature index'},
            'yaxis': {'title': 'Num of occurrences'},
        },
    }


@app.callback(
    dash.dependencies.Output('co-occurrence-graph', 'figure'),
    [dash.dependencies.Input('proglenfilter-slider', 'value')])
def update_accuracy_graph(pro_len):
    if pro_len > 1:
        cooc_matrix, feature_index = result.get_feature_co_occurences_matrix(pro_len)
        feature_index = ['f' + str(i) for i in feature_index]
        return {
            'data': [{
                'z': cooc_matrix,
                'x': feature_index,
                'y': feature_index,
                'type': 'heatmap',
                'colorscale': 'Viridis'
            }],
            'layout': {
                'title' : 'Co-occurrence of ' + str(pro_len) + ' Feature Models'
            }
        }
    return  {}


@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value')])
def update_feature_comparision_graph(xaxis_column_name, yaxis_column_name):
    xaxis_column_name = int(xaxis_column_name)
    yaxis_column_name = int(yaxis_column_name)
    type_name = ['AD', 'Normal']
    return {
        'data': [dict(
            x=X[:, int(xaxis_column_name)][y == type],
            y=X[:, int(yaxis_column_name)][y == type],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'},
            },
            name=type_name[type]
        ) for type in [0, 1]
        ],
        'layout': dict(
            xaxis={
                'title': names[xaxis_column_name],
                'type': 'linear'
            },
            yaxis={
                'title': names[yaxis_column_name],
                'type': 'linear'
            },
            hovermode='closest',
            clickmode='event+select',
            title='Two Feature Comparision'
        )
    }


@app.callback(
    Output('click-data', 'children'),
    [Input('crossfilter-indicator-scatter', 'clickData')])
def display_click_data(clickData):
    if clickData is not None:
        i = int(clickData['points'][0]['pointIndex'])
        return result.model_list[i].bestEffProgStr_


# Running server
if __name__ == "__main__":
    app.run_server(debug=True)
