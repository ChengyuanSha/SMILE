import dash
import numpy as np
import base64
import datetime
import io
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from data_processing._processing_funcs import ResultProcessing

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)

app.title = 'LGP'

# setting up get global_result class
# global_result = ResultProcessing("dataset/RuiJin_Processed.csv", "dataset/lgp_filtered.pkl")
# global_result.load_models()
# X, y, names = global_result.X, global_result.y, global_result.names
# index_list = [i for i in range(len(names))]
# available_indicators = list(zip(index_list, names))

global_result = ResultProcessing("dataset/RuiJin_Processed.csv")
X, y, names = 0, 0, 0
index_list = 0
available_indicators = 0
file_uploaded = False

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
                                    "global_result Visualization", style={"margin-top": "0px"}
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

        # File upload
        html.Div([

            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.H6('Upload pickle global_result file here'),
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                className="pretty_container six columns",
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Div(id='output-data-upload',
                     ),
            ],

        ),

        html.Div(id='tabs-content')

    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"}
)


def render_main_visualization_layout():
    return html.Div([

        render_prog_len_slider(),
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


def render_prog_len_slider():
    global_result.calculate_featureList_and_calcvariableList()
    length_list = sorted(list(set([len(i) for i in global_result.feature_list])))
    return html.Div([

        html.Div([
                html.H6('Choose Length of Feature in a Program'),

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
    prog_index, acc_scores = global_result.get_accuracy_given_length(pro_len)
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
    features, num_of_occurrences = global_result.get_occurrence_from_feature_list_given_length(pro_len)
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
        cooc_matrix, feature_index = global_result.get_feature_co_occurences_matrix(pro_len)
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
        return global_result.model_list[i].bestEffProgStr_


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'pkl' in filename:

            global global_result, X, y, names, index_list , available_indicators
            # initialize staff
            global_result = ResultProcessing("dataset/RuiJin_Processed.csv")
            global_result.load_models_directly(io.BytesIO(decoded))
            X, y, names = global_result.X, global_result.y, global_result.names
            index_list = [i for i in range(len(names))]
            available_indicators = list(zip(index_list, names))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
            ],
        )

    return html.Div([
        html.H6(filename),
        html.H6("successfully read")
        ],
    )


@app.callback([Output('output-data-upload', 'children'),
               Output('tabs-content', 'children')],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    # display read file status and update main visualization Div
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children, render_main_visualization_layout()
    else:
        return ' ',' '



# Running server
if __name__ == "__main__":
    app.run_server(debug=True)
