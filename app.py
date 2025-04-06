import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import time
import threading
import random
import os

# Initialize the Dash app with external styles
app = dash.Dash(__name__, external_stylesheets=[
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css',
    'https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap'
])
server = app.server

# Configuration
classroom_config = {
    "teacher_area": (0.3, 0.1, 0.7, 0.3),
    "board_area": (0.1, 0.1, 0.9, 0.3),
    "student_areas": [
        (0.1, 0.4, 0.3, 0.8),
        (0.3, 0.4, 0.5, 0.8),
        (0.5, 0.4, 0.7, 0.8),
        (0.7, 0.4, 0.9, 0.8)
    ]
}

students = [
    {"id": 1, "name": "Amon Muhairwe", "seat": 0},
    {"id": 2, "name": "Osen", "seat": 1},
    {"id": 3, "name": "Kip", "seat": 2},
    {"id": 4, "name": "David", "seat": 3}
]

# Global Variables
current_frame = None
frame_lock = threading.Lock()
analysis_mode = "realtime"
video_processing = False
processed_results = None
webcam = None
webcam_active = False
attention_scores = [random.uniform(0.6, 0.9) for _ in range(4)]

# Recording variables
is_recording = False
video_writer = None
recording_start_time = None
recording_dir = "recordings"
os.makedirs(recording_dir, exist_ok=True)

# Webcam Initialization
def init_webcam():
    global webcam, webcam_active
    try:
        webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if webcam.isOpened():
            webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            webcam_active = True
            return True
        return False
    except:
        return False

# Generate mock classroom frame
def generate_mock_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (240, 240, 240)
    
    # Draw classroom elements
    cv2.rectangle(frame, (384, 72), (896, 216), (200, 200, 255), -1)
    cv2.putText(frame, "Teacher", (512, 144), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    cv2.rectangle(frame, (128, 72), (1152, 216), (220, 220, 180), -1)
    cv2.putText(frame, "Board", (576, 144), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Draw student areas
    areas = [(128, 288, 384, 576), (384, 288, 640, 576), 
             (640, 288, 896, 576), (896, 288, 1152, 576)]
    
    for i, (x1, y1, x2, y2) in enumerate(areas):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 255, 200), 2)
        head_center = ((x1+x2)//2, (y1+y2)//2)
        cv2.circle(frame, head_center, 30, (255, 0, 0), -1)
        cv2.putText(frame, f"S{i+1}", (head_center[0]-15, head_center[1]+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

# Start recording function
def start_recording():
    global is_recording, video_writer, recording_start_time
    
    if not webcam_active:
        print("Cannot record - no webcam available")
        return False
    
    recording_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{recording_dir}/lecture_{recording_start_time}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        filename,
        fourcc,
        10.0,
        (1280, 720)
    )
    
    is_recording = True
    print(f"Started recording: {filename}")
    return True

# Stop recording function
def stop_recording():
    global is_recording, video_writer
    
    if video_writer is not None:
        video_writer.release()
        video_writer = None
    
    is_recording = False
    print("Stopped recording")

# Webcam feed simulation
def webcam_feed():
    global current_frame, attention_scores, webcam_active, is_recording, video_writer
    
    while analysis_mode == "realtime" and not video_processing:
        if webcam_active:
            try:
                ret, frame = webcam.read()
                if not ret:
                    print("Webcam frame read failed, switching to mock feed")
                    webcam_active = False
                    continue
                
                frame = cv2.resize(frame, (1280, 720))
                
                if is_recording and video_writer is not None:
                    video_writer.write(frame)
                
                attention_scores = [max(0, min(1, score + random.uniform(-0.05, 0.05))) for score in attention_scores]
                
                h, w = frame.shape[:2]
                for i, score in enumerate(attention_scores):
                    if score > 0.5:
                        x = int(w/2 + random.gauss(0, w/8 * (1 - score)))
                        y = int(h/3 + random.gauss(0, h/8 * (1 - score)))
                        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                    else:
                        x = random.randint(0, w)
                        y = random.randint(0, h)
                        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                
                with frame_lock:
                    current_frame = frame.copy()
            except Exception as e:
                print(f"Webcam error: {e}")
                webcam_active = False
        else:
            frame = generate_mock_frame()
            attention_scores = [max(0, min(1, score + random.uniform(-0.05, 0.05))) for score in attention_scores]
            with frame_lock:
                current_frame = frame.copy()
        
        time.sleep(0.1)
    
    if video_writer is not None:
        video_writer.release()

# Initialize webcam and start thread
if init_webcam():
    print("Webcam initialized successfully")
else:
    print("Webcam initialization failed, using mock feed")
webcam_thread = threading.Thread(target=webcam_feed, daemon=True)
webcam_thread.start()

# Custom CSS styles
custom_styles = {
    'header': {
        'background': 'linear-gradient(135deg, #3498db 0%, #2c3e50 100%)',
        'color': 'white',
        'padding': '1.5rem',
        'borderRadius': '0 0 10px 10px',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.1)',
        'marginBottom': '2rem'
    },
    'card': {
        'background': 'white',
        'borderRadius': '10px',
        'boxShadow': '0 4px 15px rgba(0,0,0,0.05)',
        'padding': '1.5rem',
        'marginBottom': '1.5rem',
        'border': '1px solid rgba(0,0,0,0.05)'
    },
    'button': {
        'primary': {
            'background': 'linear-gradient(135deg, #3498db 0%, #2980b9 100%)',
            'border': 'none',
            'color': 'white',
            'padding': '0.75rem 1.5rem',
            'borderRadius': '50px',
            'fontWeight': '600',
            'cursor': 'pointer',
            'boxShadow': '0 4px 10px rgba(52, 152, 219, 0.3)',
            'transition': 'all 0.3s ease'
        },
        'danger': {
            'background': 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)',
            'border': 'none',
            'color': 'white',
            'padding': '0.75rem 1.5rem',
            'borderRadius': '50px',
            'fontWeight': '600',
            'cursor': 'pointer',
            'boxShadow': '0 4px 10px rgba(231, 76, 60, 0.3)',
            'transition': 'all 0.3s ease'
        },
        'success': {
            'background': 'linear-gradient(135deg, #2ecc71 0%, #27ae60 100%)',
            'border': 'none',
            'color': 'white',
            'padding': '0.75rem 1.5rem',
            'borderRadius': '50px',
            'fontWeight': '600',
            'cursor': 'pointer',
            'boxShadow': '0 4px 10px rgba(46, 204, 113, 0.3)',
            'transition': 'all 0.3s ease'
        },
        'secondary': {
            'background': 'linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%)',
            'border': 'none',
            'color': 'white',
            'padding': '0.75rem 1.5rem',
            'borderRadius': '50px',
            'fontWeight': '600',
            'cursor': 'pointer',
            'boxShadow': '0 4px 10px rgba(149, 165, 166, 0.3)',
            'transition': 'all 0.3s ease'
        }
    },
    'indicator': {
        'good': {
            'background': 'linear-gradient(135deg, #2ecc71 0%, #27ae60 100%)',
            'color': 'white',
            'padding': '0.5rem 1rem',
            'borderRadius': '50px',
            'fontWeight': '600',
            'display': 'inline-block'
        },
        'warning': {
            'background': 'linear-gradient(135deg, #f39c12 0%, #e67e22 100%)',
            'color': 'white',
            'padding': '0.5rem 1rem',
            'borderRadius': '50px',
            'fontWeight': '600',
            'display': 'inline-block'
        },
        'danger': {
            'background': 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)',
            'color': 'white',
            'padding': '0.5rem 1rem',
            'borderRadius': '50px',
            'fontWeight': '600',
            'display': 'inline-block'
        }
    }
}

# Dashboard Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("MS-GESCAM Classroom Analytics", style={
                'marginBottom': '0.5rem',
                'fontWeight': '700',
                'fontSize': '2.2rem'
            }),
            html.P("Multi-Stream Gaze Estimation for Classroom Attention Measurement", style={
                'marginTop': '0',
                'fontSize': '1.1rem',
                'opacity': '0.9'
            }),
            html.Div([
                html.Span([
                    html.I(className="fas fa-book", style={'marginRight': '8px'}),
                    "18-799-RW Applied Computer Vision"
                ], style={
                    'backgroundColor': 'rgba(255,255,255,0.2)',
                    'padding': '0.5rem 1rem',
                    'borderRadius': '50px',
                    'fontSize': '0.9rem'
                })
            ], style={'marginTop': '1rem'})
        ], style={'maxWidth': '1200px', 'margin': '0 auto'})
    ], style=custom_styles['header']),
    
    # Main content
    html.Div([
        # Left sidebar
        html.Div([
            # Analysis Mode Card
            html.Div([
                html.H4("Analysis Mode", style={
                    'marginBottom': '1rem',
                    'color': '#2c3e50',
                    'display': 'flex',
                    'alignItems': 'center'
                }),
                dcc.RadioItems(
                    id='analysis-mode',
                    options=[
                        {'label': html.Span([
                            html.I(className="fas fa-video", style={'marginRight': '10px'}),
                            "Real-time Monitoring"
                        ], style={'display': 'flex', 'alignItems': 'center'}), 
                        'value': 'realtime'},
                        {'label': html.Span([
                            html.I(className="fas fa-upload", style={'marginRight': '10px'}),
                            "Upload Recording"
                        ], style={'display': 'flex', 'alignItems': 'center'}), 
                        'value': 'upload'}
                    ],
                    value='realtime',
                    labelStyle={'display': 'block', 'marginBottom': '15px'},
                    inputStyle={'marginRight': '10px'}
                )
            ], style=custom_styles['card']),
            
            html.Div(id='upload-section', style={'display': 'none'}),
            
            # Recording Card
            html.Div([
                html.H4("Lecture Recording", style={
                    'marginBottom': '1rem',
                    'color': '#2c3e50',
                    'display': 'flex',
                    'alignItems': 'center'
                }),
                html.Div(id='recording-status', style={
                    'background': 'linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%)',
                    'color': 'white',
                    'padding': '0.75rem',
                    'borderRadius': '8px',
                    'marginBottom': '1rem',
                    'textAlign': 'center',
                    'fontWeight': '600'
                }),
                html.Div([
                    html.Button(
                        html.Span([
                            html.I(className="fas fa-circle", style={'marginRight': '8px'}),
                            "Start Recording"
                        ]),
                        id='start-recording',
                        n_clicks=0,
                        style=custom_styles['button']['danger']
                    ),
                    html.Button(
                        html.Span([
                            html.I(className="fas fa-stop", style={'marginRight': '8px'}),
                            "Stop Recording"
                        ]),
                        id='stop-recording',
                        n_clicks=0,
                        style={**custom_styles['button']['secondary'], 'marginLeft': '10px'}
                    )
                ], style={'display': 'flex', 'justifyContent': 'center'})
            ], style=custom_styles['card']),
            
            # Overall Attention Card
            html.Div([
                html.H4("Overall Attention", style={
                    'marginBottom': '1rem',
                    'color': '#2c3e50',
                    'display': 'flex',
                    'alignItems': 'center'
                }),
                html.Div(id='overall-attention-score', style={
                    'fontSize': '2.5rem',
                    'fontWeight': '700',
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'marginBottom': '1rem',
                    'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
                    'padding': '1.5rem',
                    'borderRadius': '10px',
                    'boxShadow': 'inset 0 4px 15px rgba(0,0,0,0.05)'
                }),
                dcc.Graph(
                    id='student-scores-chart',
                    style={'height': '250px'},
                    config={'displayModeBar': False}
                )
            ], style=custom_styles['card'])
        ], className="three columns", style={'paddingRight': '20px'}),
        
        # Main content area
        html.Div([
            # Video Display Card
            html.Div([
                html.Div([
                    html.H3(id='view-title', style={
                        'marginBottom': '0.5rem',
                        'color': '#2c3e50'
                    }),
                    html.P("Live classroom monitoring and analysis", style={
                        'marginTop': '0',
                        'color': '#7f8c8d',
                        'marginBottom': '1.5rem'
                    })
                ], style={'marginBottom': '1rem'}),
                
                html.Div([
                    html.Button(
                        html.Span([
                            html.I(className="fas fa-eye", style={'marginRight': '8px'}),
                            "Raw Feed"
                        ]),
                        id='display-mode-raw',
                        n_clicks=0,
                        style={**custom_styles['button']['secondary'], 'marginRight': '10px'}
                    ),
                    html.Button(
                        html.Span([
                            html.I(className="fas fa-fire", style={'marginRight': '8px'}),
                            "Heatmap"
                        ]),
                        id='display-mode-heatmap',
                        n_clicks=0,
                        style={**custom_styles['button']['danger'], 'marginRight': '10px'}
                    ),
                    html.Button(
                        html.Span([
                            html.I(className="fas fa-chart-line", style={'marginRight': '8px'}),
                            "Engagement"
                        ]),
                        id='display-mode-engagement',
                        n_clicks=0,
                        style=custom_styles['button']['success']
                    )
                ], style={'marginBottom': '1.5rem'}),
                
                html.Div(
                    id='video-display',
                    style={
                        'borderRadius': '10px',
                        'overflow': 'hidden',
                        'boxShadow': '0 4px 20px rgba(0,0,0,0.1)',
                        'height': '400px',
                        'background': '#f8f9fa',
                        'border': '1px solid rgba(0,0,0,0.05)'
                    }
                )
            ], style={
                **custom_styles['card'],
                'padding': '2rem'
            }),
            
            # Engagement Alert
            html.Div(id='engagement-alert', style={'marginBottom': '1.5rem'}),
            
            # Charts Row
            html.Div([
                html.Div([
                    html.Div([
                        html.H4("Engagement Overview", style={
                            'marginBottom': '1rem',
                            'color': '#2c3e50'
                        }),
                        dcc.Graph(
                            id='engagement-chart', 
                            style={'height': '250px'},
                            config={'displayModeBar': False}
                        )
                    ], style=custom_styles['card'])
                ], className="six columns"),
                
                html.Div([
                    html.Div([
                        html.H4("Attention Trend", style={
                            'marginBottom': '1rem',
                            'color': '#2c3e50'
                        }),
                        dcc.Graph(
                            id='attention-trend', 
                            style={'height': '250px'},
                            config={'displayModeBar': False}
                        )
                    ], style=custom_styles['card'])
                ], className="six columns")
            ], className="row")
        ], className="six columns", style={'paddingRight': '20px'}),
        
        # Right sidebar
        html.Div([
            html.Div([
                html.H4("Student Analytics", style={
                    'marginBottom': '1.5rem',
                    'color': '#2c3e50',
                    'display': 'flex',
                    'alignItems': 'center'
                }),
                dcc.Dropdown(
                    id='student-selector',
                    options=[{'label': s['name'], 'value': s['id']} for s in students],
                    value=students[0]['id'],
                    clearable=False,
                    style={'marginBottom': '1.5rem'}
                ),
                html.Div(id='student-details')
            ], style=custom_styles['card'])
        ], className="three columns")
    ], className="row", style={
        'maxWidth': '1400px',
        'margin': '0 auto',
        'padding': '0 20px'
    }),
    
    # Hidden stores
    dcc.Store(id='historical-data', data={'timestamps': [], 'scores': []}),
    dcc.Store(id='session-data'),
    dcc.Store(id='display-mode', data='heatmap'),
    dcc.Interval(id='live-update', interval=1000)
], style={
    'fontFamily': '"Open Sans", sans-serif',
    'backgroundColor': '#f8f9fa',
    'minHeight': '100vh',
    'paddingBottom': '2rem'
})

# Callbacks
@app.callback(
    [Output('upload-section', 'style'),
     Output('upload-section', 'children'),
     Output('live-update', 'disabled'),
     Output('view-title', 'children')],
    [Input('analysis-mode', 'value')]
)
def update_mode(mode):
    global analysis_mode
    analysis_mode = mode
    
    if mode == "upload":
        upload_section = html.Div([
            dcc.Upload(
                id='upload-video',
                children=html.Div([
                    html.Div([
                        html.I(className="fas fa-cloud-upload-alt", style={'fontSize': '2rem', 'marginBottom': '1rem'}),
                        html.P("Drag and Drop or Click to Select Video", style={'marginBottom': '0'})
                    ], style={'textAlign': 'center', 'padding': '2rem'})
                ]),
                style={
                    'width': '100%',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '8px',
                    'textAlign': 'center',
                    'marginBottom': '1.5rem',
                    'cursor': 'pointer',
                    'borderColor': '#3498db',
                    'background': 'rgba(52, 152, 219, 0.05)',
                    'transition': 'all 0.3s ease'
                },
                multiple=False
            ),
            html.Div(id='upload-status')
        ])
        return {'display': 'block'}, upload_section, True, "Lecture Recording Analysis"
    else:
        return {'display': 'none'}, None, False, "Live Classroom Monitoring"

@app.callback(
    Output('display-mode', 'data'),
    [Input('display-mode-raw', 'n_clicks'),
     Input('display-mode-heatmap', 'n_clicks'),
     Input('display-mode-engagement', 'n_clicks')],
    [State('display-mode', 'data')]
)
def update_display_mode(raw_clicks, heatmap_clicks, engagement_clicks, current_mode):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_mode
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    return button_id.split('-')[-1]

# Recording control callback
@app.callback(
    [Output('recording-status', 'children'),
     Output('recording-status', 'style')],
    [Input('start-recording', 'n_clicks'),
     Input('stop-recording', 'n_clicks')]
)
def control_recording(start_clicks, stop_clicks):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return "Ready to record", {
            'background': 'linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%)'
        }
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-recording':
        success = start_recording()
        if success:
            return "● Recording...", {
                'background': 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)'
            }
        else:
            return "Recording failed", {
                'background': 'linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%)'
            }
    else:
        stop_recording()
        return "Recording saved", {
            'background': 'linear-gradient(135deg, #2ecc71 0%, #27ae60 100%)'
        }

@app.callback(
    [Output('video-display', 'children'),
     Output('engagement-chart', 'figure'),
     Output('attention-trend', 'figure'),
     Output('engagement-alert', 'children'),
     Output('historical-data', 'data'),
     Output('session-data', 'data'),
     Output('overall-attention-score', 'children'),
     Output('student-scores-chart', 'figure')],
    [Input('live-update', 'n_intervals'),
     Input('display-mode', 'data'),
     Input('analysis-mode', 'value')],
    [State('historical-data', 'data'),
     State('session-data', 'data')]
)
def update_dashboard(n, display_mode, mode, historical_data, session_data):
    global attention_scores
    
    # Get current frame
    with frame_lock:
        frame = current_frame.copy() if current_frame is not None else generate_mock_frame()
    
    avg_score = np.mean(attention_scores)
    
    # Update historical data
    timestamp = datetime.now().strftime("%H:%M:%S")
    if len(historical_data['timestamps']) == 0 or historical_data['timestamps'][-1] != timestamp:
        historical_data['timestamps'].append(timestamp)
        historical_data['scores'].append(avg_score)
        if len(historical_data['timestamps']) > 20:
            historical_data['timestamps'].pop(0)
            historical_data['scores'].pop(0)
    
    # Generate display based on mode
    if display_mode == 'raw':
        video_display = generate_live_view(frame)
    elif display_mode == 'heatmap':
        video_display = generate_heatmap_view(frame)
    else:
        video_display = generate_engagement_view(frame, attention_scores)
    
    # Generate charts
    engagement_fig = generate_engagement_figure(avg_score)
    trend_fig = generate_trend_figure(historical_data)
    alert = generate_alert(avg_score)
    
    # Student scores chart
    student_fig = px.bar(
        x=[s['name'] for s in students],
        y=attention_scores,
        labels={'x': 'Student', 'y': 'Attention Score'},
        range_y=[0, 1],
        color=[s['name'] for s in students],
        color_discrete_sequence=['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    )
    student_fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': '"Open Sans", sans-serif'}
    )
    student_fig.update_traces(marker_line_width=0)
    
    return (video_display, engagement_fig, trend_fig, alert, 
            historical_data, session_data, 
            f"{avg_score:.0%}", student_fig)

@app.callback(
    Output('student-details', 'children'),
    [Input('student-selector', 'value')]
)
def update_student_details(student_id):
    student = next(s for s in students if s['id'] == student_id)
    score = attention_scores[student['seat']]
    
    return html.Div([
        html.Div([
            html.Div([
                html.Img(
                    src=f"https://ui-avatars.com/api/?name={student['name'].replace(' ', '+')}&background=random&size=100",
                    style={
                        'width': '80px',
                        'height': '80px',
                        'borderRadius': '50%',
                        'objectFit': 'cover',
                        'marginBottom': '1rem',
                        'boxShadow': '0 4px 10px rgba(0,0,0,0.1)'
                    }
                ),
                html.H4(student['name'], style={
                    'textAlign': 'center',
                    'margin': '0 0 0.5rem 0',
                    'color': '#2c3e50'
                }),
                html.Div(f"Seat {student['seat'] + 1}", style={
                    'textAlign': 'center',
                    'color': '#7f8c8d',
                    'marginBottom': '1.5rem',
                    'fontSize': '0.9rem'
                })
            ], style={'textAlign': 'center'})
        ], style={'marginBottom': '1.5rem'}),
        
        html.Div([
            html.Div("Current Attention", style={
                'fontSize': '0.9rem',
                'color': '#7f8c8d',
                'textAlign': 'center',
                'marginBottom': '0.5rem'
            }),
            html.Div([
                html.Div(f"{score:.0%}", style={
                    'fontSize': '2rem',
                    'fontWeight': '700',
                    'textAlign': 'center',
                    'color': '#2c3e50'
                }),
                html.Div([
                    html.I(className="fas fa-arrow-up" if score > 0.6 else "fas fa-arrow-down", style={
                        'marginRight': '5px',
                        'color': '#2ecc71' if score > 0.6 else '#e74c3c'
                    }),
                    "2% from last 5 min" if score > 0.6 else "5% from last 5 min"
                ], style={
                    'textAlign': 'center',
                    'color': '#7f8c8d',
                    'fontSize': '0.8rem'
                })
            ], style={
                'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
                'padding': '1.5rem',
                'borderRadius': '10px',
                'marginBottom': '1.5rem',
                'boxShadow': 'inset 0 4px 15px rgba(0,0,0,0.05)'
            })
        ]),
        
        dcc.Graph(
            figure={
                'data': [{
                    'x': ['Attention'], 
                    'y': [score], 
                    'type': 'bar',
                    'marker': {'color': '#3498db'}
                }],
                'layout': {
                    'yaxis': {'range': [0, 1]},
                    'margin': {'l': 40, 'r': 40, 't': 30, 'b': 30},
                    'height': 200,
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                    'paper_bgcolor': 'rgba(0,0,0,0)'
                }
            },
            config={'displayModeBar': False}
        )
    ])

# Visualization functions
def generate_live_view(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig = px.imshow(frame_rgb)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return dcc.Graph(
        figure=fig,
        style={'height': '100%'},
        config={'staticPlot': True}
    )

def generate_heatmap_view(frame):
    h, w = frame.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    for _ in range(4):
        x = random.randint(0, w)
        y = random.randint(0, h)
        cv2.circle(heatmap, (x, y), 50, 255, -1)
    heatmap = cv2.GaussianBlur(heatmap, (101, 101), 0)
    heatmap = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    fig = px.imshow(overlay_rgb)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return dcc.Graph(
        figure=fig,
        style={'height': '100%'},
        config={'staticPlot': True}
    )

def generate_engagement_view(frame, scores):
    viz_frame = frame.copy()
    h, w = frame.shape[:2]
    for i, score in enumerate(scores):
        area = classroom_config["student_areas"][i]
        x = int(w * (area[0] + area[2]) / 2)
        y = int(h * (area[1] + area[3]) / 2)
        color = (0, 255, 0) if score > 0.7 else (0, 255, 255) if score > 0.4 else (0, 0, 255)
        cv2.circle(viz_frame, (x, y), 30, color, -1)
        cv2.putText(viz_frame, f"{score:.0%}", (x-15, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    viz_frame_rgb = cv2.cvtColor(viz_frame, cv2.COLOR_BGR2RGB)
    fig = px.imshow(viz_frame_rgb)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return dcc.Graph(
        figure=fig,
        style={'height': '100%'},
        config={'staticPlot': True}
    )

def generate_engagement_figure(score):
    fig = px.pie(
        values=[score, 1-score],
        names=["Engaged", "Not Engaged"],
        hole=0.4,
        color_discrete_sequence=['#2ecc71', '#e74c3c']
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': '"Open Sans", sans-serif'}
    )
    return fig

def generate_trend_figure(historical_data):
    if len(historical_data['timestamps']) > 0:
        df = pd.DataFrame({
            'Time': historical_data['timestamps'],
            'Engagement': historical_data['scores']
        })
        fig = px.line(
            df,
            x='Time',
            y='Engagement',
            markers=True,
            line_shape='spline'
        )
        fig.update_traces(
            line=dict(color='#3498db', width=3),
            marker=dict(color='#3498db', size=8)
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(
                range=[0, 1],
                gridcolor='rgba(0,0,0,0.05)',
                zeroline=False
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': '"Open Sans", sans-serif'},
            xaxis=dict(
                showgrid=False,
                zeroline=False
            )
        )
    else:
        fig = px.line()
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(range=[0, 1]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(
                text="No data yet",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16)
            )]
        )
    return fig

def generate_alert(score):
    if score < 0.4:
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle", style={'marginRight': '10px'}),
                "Low Engagement! Consider changing activities."
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center'
            })
        ], style={
            **custom_styles['indicator']['danger'],
            'padding': '1rem',
            'width': '100%'
        })
    elif score < 0.6:
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-circle", style={'marginRight': '10px'}),
                "Moderate Engagement - Some students may need support"
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center'
            })
        ], style={
            **custom_styles['indicator']['warning'],
            'padding': '1rem',
            'width': '100%'
        })
    return html.Div([
        html.Div([
            html.I(className="fas fa-check-circle", style={'marginRight': '10px'}),
            "Good Engagement Level Maintained"
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'center'
        })
    ], style={
        **custom_styles['indicator']['good'],
        'padding': '1rem',
        'width': '100%'
    })

if __name__ == '__main__':
    app.run_server(debug=True)