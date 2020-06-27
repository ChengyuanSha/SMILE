import dash
import numpy as np
import base64
import datetime
import os
import copy
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

# global variable will cause problems in multi-user web, fix later
original_result = ResultProcessing("dataset/RuiJin_Processed.csv")
global_result = 0
X, y, names = 0, 0, 0
index_list = 0
available_indicators = 0

server = app.server

def render_webpage_title():
    return html.Div(
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
    )


# need to suppress bc using dynamic tabs
app.config['suppress_callback_exceptions'] = True

app.layout = html.Div(
    [
        # --- headline ---
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        # title
        render_webpage_title(),
        # File upload
        html.Div([

            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.H6('Upload pickle result file here'),
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                className="pretty_container six columns",
                # Allow multiple files to be uploaded
                multiple=True
            ),

            html.A(
                html.Button(
                    'Download sample pickle result data',
                    id='speck-file-download',
                    className='control-download'
                ),
                href=os.path.join('assets', 'sample_data', 'lgp_sample.pkl'),
                download='lgp_sample.pkl'
            ),

            html.Div(id='output-data-upload',
                     ),
        ],
        ),
        # main page
        html.Div(id='main_visualization_content')

    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"}
)


def render_main_visualization_layout():
    return html.Div([
        # ---------- sliders/filters -----------
        html.Div([

            html.Div([
                html.H6('Choose testing set accuracy threshold to filter'),

                dcc.Slider(
                    id='testing-acc-filter-slider',
                    min=0,
                    max=100,
                    value=90,
                    marks={str(num): str(num) + '%' for num in range(10, 100, 10)},
                    step=1,
                    updatemode='drag'
                ),
                html.Div(id='updatemode-output-testing-acc', style={'margin-top': 20})

            ],
                className="pretty_container six columns",
            ),

            html.Div([
                html.H6('Choose number of Feature in a Program'),

                # dcc.Slider(id='prog-len-filter-slider',marks={int(each_len): 'Len' + str(each_len) for each_len in [1,2]}),
                dcc.RadioItems(id='prog-len-filter-slider', labelStyle={'display': 'inline-block'}),
                # html.Div([dcc.Slider(id='slider')], id='slider-keeper'),  # dummy slider
                # html.Div(id='prog-len-filter-slider'),
                html.Div(id='updatemode-output-proglenfilter', style={'margin-top': 20})
            ],
                className="pretty_container six columns",
            ),

        ],
            style={'align-items': 'center', 'justify-content': 'center'},
            className="row container-display",
        ),

        html.Div(
            [
                html.Div(
                    [html.H6(str(len(original_result.model_list))), html.P("Original Model Count")],
                    id="Original Model Count",
                    className="mini_container",
                ),
                html.Div(
                    [html.H6(id="filtered_by_accuracy_text"), html.P("Model Count After Filtered by Testing Set Accuracy")],
                    id="gas",
                    className="mini_container",
                ),
                html.Div(
                    [html.H6(id="filtered_by_len_text"), html.P("Model Count After Filtered by Accuracy and Number of Feature")],
                    id="oil",
                    className="mini_container",
                ),
                html.Div(
                    [html.H6(id="waterText"), html.P("Maybe some extra info?")],
                    id="water",
                    className="mini_container",
                ),
            ],
            className="row container-display",
        ),
        # ------------------   first 2 graphs in website  --------------
        html.Div([
            html.Div([
                dcc.Graph(
                    id='filtered-occurrences-scatter',
                )
            ],
                id="left-column",
                className="pretty_container six columns",
            ),

            html.Div([
                dcc.Graph(
                    id='filtered-accuracy-scatter',
                )
            ],
                id="right-column",
                className="pretty_container six columns",
            ),
        ],
            className="row flex-display",
        ),
        #     ------------------   model visualization  --------------
        html.Div([
            html.Div([
                dcc.Markdown("""
                    **Click Models On Model Accuracy Scatter Plot**  
                    Detailed Model Info:
                """),

                html.Pre(id='model-click-data',
                         style={
                             'border': 'thin lightgrey solid',
                             'overflowX': 'scroll'
                         }),
            ],
            className="pretty_container six columns",)
        ],
        className="row flex-display",
        ),

        # row 2 selectors in website
        html.Div([
            html.Div([
                dcc.Markdown('''
                    **Click on co-occurrence heat map to see two feature distribution in original data.**    
                    Or manually choose X axis / Y axis for two distribution graph on dropdown manual.
                '''),

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
            className="row flex-display",
            style={'align-items': 'center', 'justify-content': 'center'},
        ),

        # 4 row Two Feature Comparision in the website
        html.Div([
            # Two Feature Co-occurrence in website
            html.Div([
                html.Div(
                    html.H6('Two Feature Co-occurrence Analysis (Only For 2+ Features)')
                ),

                html.Div(
                    dcc.Graph(
                        id='co-occurrence-graph'
                    )
                )

            ],
                className="pretty_container four columns"
            ),

            # 2 feature scatter plot, update based on x, y filter, see the callback
            html.Div([
                dcc.Graph(
                    id='crossfilter-indicator-scatter',
                    # hoverData={'points': [{'customdata': 'Japan'}]}
                )
            ],
                className="pretty_container seven columns",
            ),
        ],
        className="row flex-display"
        ),
    ])

@app.callback(
    Output('filtered-accuracy-scatter', 'figure'),
    [Input('filtered-occurrences-scatter', 'clickData'),
     Input('prog-len-filter-slider', 'value')])
def update_accuracy_graph_based_on_clicks(clickData, prog_len):
    if clickData is not None:
        feature_num = int(clickData['points'][0]['x'][1:]) # extract data from click
        m_index = global_result.get_index_of_models_given_feature_and_length(feature_num, prog_len)
        testing_acc = [global_result.model_list[i].testingAccuracy for i in m_index]
        m_index = ['m' + str(i) for i in m_index]
        return {
                'data': [
                    {'x': m_index,
                     'y': testing_acc,
                     'mode':'markers',
                     'marker': {'size': 3}
                    },
                ],
                'layout': {
                    'title': 'Model accuracy containing feature ' + str(feature_num) + ' with' + str(prog_len) + ' features',
                    'xaxis': {'title': 'Program feature index'},
                    'yaxis': {'title': 'Num of occurrences'},
                    'clickmode': 'event+select'
                }
        }
    return {  'layout': {
                    'title': 'No graph in given selection. Click on Occurrence graph.'
                }}


@app.callback(
    [Output('filtered-occurrences-scatter', 'figure'),
     Output("filtered_by_accuracy_text", "children"),
     Output("filtered_by_len_text", "children")],
    [Input('prog-len-filter-slider', 'value'),
     ])
def update_occurrence_graph(pro_len):
    # global_result.model_list = [i for i in original_result.model_list if
    #                             float(i.testingAccuracy) >= ((testing_acc) / 100)]
    global_result.calculate_featureList_and_calcvariableList()
    features, num_of_occurrences, cur_feature_num = global_result.get_occurrence_from_feature_list_given_length(pro_len)
    hover_text = [names[i] for i in features]
    features = ['f' + str(i) for i in features]
    return {
               'data': [{
                    'x': features,
                    'y': num_of_occurrences,
                    'type': 'bar',
                    'hoverinfo': 'text',
                    'text': hover_text
               }],
               'layout': {
                   'title': 'Occurrences of Features of ' + str(pro_len) + ' Feature Models',
                   'xaxis': {'title': 'Program feature index'},
                   'yaxis': {'title': 'Num of occurrences'},
               },
           }, len(global_result.model_list), cur_feature_num



@app.callback(
    Output('co-occurrence-graph', 'figure'),
    [Input('prog-len-filter-slider', 'value')])
def update_co_occurrence_graph(pro_len):
    if pro_len > 1:
        cooc_matrix, feature_index = global_result.get_feature_co_occurences_matrix(pro_len)
        hover_text = []
        for yi, yy in enumerate(feature_index):
            hover_text.append([])
            for xi, xx in enumerate(feature_index):
                hover_text[-1].append('X: {}<br />Y: {}<br />Count: {}'.format(names[int(xx)], names[int(yy)], cooc_matrix[xi, yi]))
        feature_index = ['f' + str(i) for i in feature_index]
        return {
            'data': [{
                'z': cooc_matrix,
                'x': feature_index,
                'y': feature_index,
                'type': 'heatmap',
                'colorscale': 'Viridis',
                'hoverinfo': 'text',
                'text': hover_text
            }],
            'layout': {
                'title': 'Co-occurrence of ' + str(pro_len) + ' Feature Models',
                #'margin': dict(l=20, r=20, t=20, b=20)
            }
        }
    return {}


@app.callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    [Input('crossfilter-xaxis-column', 'value'),
     Input('crossfilter-yaxis-column', 'value'),
     Input('co-occurrence-graph', 'clickData')])
def update_feature_comparision_graph_using_filters(xaxis_column_index, yaxis_column_index, co_click_data):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'crossfilter-xaxis-column' or trigger_id == 'crossfilter-yaxis-column':
        xaxis_column_index = int(xaxis_column_index)
        yaxis_column_index = int(yaxis_column_index)
    elif trigger_id == 'co-occurrence-graph':
        xaxis_column_index = int(co_click_data['points'][0]['x'][1:])
        yaxis_column_index = int(co_click_data['points'][0]['y'][1:])
    type_name = ['AD', 'Normal']
    return {
        'data': [dict(
            x=X[:, int(xaxis_column_index)][y == type],
            y=X[:, int(yaxis_column_index)][y == type],
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
                'title': names[int(xaxis_column_index)],
                'type': 'linear'
            },
            yaxis={
                'title': names[int(yaxis_column_index)],
                'type': 'linear'
            },
            hovermode='closest',
            clickmode='event+select',
            title='Two Feature Comparision'
        )
    }

@app.callback(
    Output('model-click-data', 'children'),
    [Input('filtered-accuracy-scatter', 'clickData')])
def update_model_click_data(clickData):
    if clickData is not None:
        i = int(clickData['points'][0]['x'][1:])
        return global_result.convert_program_str_repr(global_result.model_list[i])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'pkl' in filename:
            global original_result, global_result, X, y, names, index_list, available_indicators
            # initialize staff
            original_result = ResultProcessing("dataset/RuiJin_Processed.csv")
            original_result.load_models_directly(io.BytesIO(decoded))
            X, y, names = original_result.X, original_result.y, original_result.names
            index_list = [i for i in range(len(names))]
            available_indicators = list(zip(index_list, names))
            global_result = copy.deepcopy(original_result)
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
               Output('main_visualization_content', 'children')],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_file_output(list_of_contents, list_of_names, list_of_dates):
    # display read file status and update main visualization Div
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children, render_main_visualization_layout()
    else:
        return ' ', ' '


@app.callback(Output('updatemode-output-testing-acc', 'children'),
              [Input('testing-acc-filter-slider', 'value')])
def update_tesing_filter_value(value):
    return str(value) + "% is used to filter models"


@app.callback(Output('updatemode-output-proglenfilter', 'children'),
              [Input('prog-len-filter-slider', 'value')])
def display_program_length_filter_value(value):
    # program length filter --> effective features text display
    return "Models with " + str(value) + " effective features are used"

@app.callback(Output('prog-len-filter-slider', 'options'),
              [Input('testing-acc-filter-slider', 'value')])
def set_prog_len_radiobutton(testing_acc):
    global global_result  # need to update global_result
    global_result.model_list = [i for i in original_result.model_list if
                                float(i.testingAccuracy) >= ((testing_acc) / 100)]
    global_result.calculate_featureList_and_calcvariableList()
    length_list = sorted(list(set([len(i) for i in global_result.feature_list])))
    return [{'label': str(i) , 'value': i} for i in length_list]


@app.callback( Output('prog-len-filter-slider', 'value'),
    [Input('prog-len-filter-slider', 'options')])
def set_prog_len_value(available_options):
    return available_options[0]['value']


# Running server
if __name__ == "__main__":
    app.run_server(debug=True)