import dash
import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy , timedelta
import yfinance as yf
import 
import ccxt

from omnipattern_decoder import OmniPatternDecoder

# Initialize OmniPatternDecoder
decoder = OmniPatternDecoder()

# Initialize the Dash app
app = dash.Dash(__name__)

# Flag variable to ensure trading functionality is triggered only once
trading_triggered 

# App layout
app.layout = html.Div([
    html.H1("OmniPattern Cosmic Market Decoder Dashboard"),
    
    html.Div([
        html."),
            dcc.Dropdown(
                id='market-dropdown',
                options=[
                    {'label': 'Gold Futures', 'value': 'GC=F'},
                    {'label': 'Silver Futures', 'value': 'SI=F'},
                    {'label': 'Bitcoin USD', 'value': 'BTC-USD'},
                    {'label': 'S&P 500', 'value': '^GSPC'},
                    {'label': 'Nasdaq', 'value': '^IXIC'}
                ],
                value='GC=F'
            ),
            
            html.H3("Time Period"),
            dcc.DatePickerRange(
                id='da',
                start_date=(.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-
            ),
            
            html.Button('Update Analysis', id='update-button', n_clicks=0),
            html.Button('Start Trading', id='trade-button', n_clicks=0),
            html.Button('Trigger Trading', id='trigger-trade-button', n_clicks=0)
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        html.Div([
            html.H3("Current Astronomical Positions"),
            html.Div(id='astro-positions'),
            
    ]),
    
    html.Div([
        dcc.Graph(id='price-chart'),
        dcc.Graph(id='pattern-chart'),
        dcc.Graph(id='cycle-chart')
    ]),
    
    dcc.Interval(
        id='interval-component',
        interval=1,  # update every millisecond
        n_intervals=0
    )
])

# Callback to update current astronomical positions
@app.callback(
    Output('astro-positions', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_astro_positions(n):
    now = datetime.now()
    ephem_date = ephem.Date(now)
    
    # Calculate positions
    moon = ephem.Moon(ephem_date)
    moon.compute(ephem_date)
    moon_p
    neptune = ephem.Neptune(ephem_date)
    neptune.compute(ephem_date)
    neptune_pos = float(neptune.hlong) * 180.0 / np.pi
    
    sun = ephem.Sun(ephem_date)
    sun.compute(ephem_date)
    sun_pos = float(sun.hlong) * 180.0 / np.pi
    
    # Determine zodiac si', 'Pisces'
    ]
    
    mars_sign = zodiac_signs[int(mars_pos / 30)]
    neptune_sign = zodiac_signs[int(neptune_pos / 30)]
    sun_sign = zodiac_signs[int(sun_pos / 30)]
    
    # Calculate angular distance between Sun and Neptune
    angle_diff = abs(neptune_pos - sun_pos)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    neptune_aspect = "Square" if 85 <= angle_diff <= 95 else "None"
    if 0 <= angle_diff <= 10 or 350 <= angle_diff <= 360:
        neptune_aspect = "Conjunction"
    elif 170 <= angle_diff <= 190:
        neptune_aspect = "Opposition"
    
    return html.Div([
        html.P(f"Current Date/Time: {now.strftime('%Y-%m-%d %H:%M:%S')}"),
        html.P(f"Moon Phase: {moon_phase:.1f}% ({('New Moon' if moon_phase < 10    html.P(f"Sun Position: {sun_pos:.1f}° in {sun_sign}"),
        html.P(f"Neptune-Sun Aspect: {neptune_aspect} ({angle_diff:.1f}°)")
    ])

# Callback to update analysis and charts
@app.callback(
    [Output('price-chart', 'figure'),
     Output('pattern-chart', 'figure'),
     Output('cycle-chart', 'figure'),
     Output('detected-patterns', 'children')],
    [Input('update-button', 'n_clicks')],
    [dash.dependencies.State('market-dropdown', 'value'),
     dash.dependencies.State('date-picker-range', 'start_date'),
     dash.dependencies.State('date-picker-range', 'end_date')]
)
def update_analysis(n_clicks, market, start_date, end_date):
    if n_clicks == 0:
        # Return empty figures on initial load
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data loaded yet. Click 'Update Analysis'.")
        return empty_fig, empty_fig, empty_fig, "No patterns detected yet."
    
    # Run analysis
    results = decoder.run_analysis(market, start_date, end_date)
    
    # Price chart
    price_fig = go.Figure()
    price_fig.add_trace(go.Candlestick(
        x=results['price_data'].index,
        open=results['price_data']['Open'],
        high=results['price_data']['High'],
        low=results['price_data']['Low'],
        close=results['price_data']['Close'],
        name='Price'
    ))
    
    # Add 9-candle pattern markers
    if not results['nine_candle_patterns'].empty:
        price_fig.add_trace(go.Scatter(
            x=results['nine_candle_patterns']['date'],'
        ))
    
    price_fig.update_layout(title=f"{market} Price Chart with Detected Patterns")
    
    # Pattern chart
    pattern_fig = go.Figure()
    
    if not results['mars_fire_analysis'].empty:
        pattern_fig.add_trace(go.Bar(
            x=results['mars_fire_analysis']['date'],
            y=results['mars_fire_analysis']['price_change'],
            name='Mars in Fire Signs'
        ))
    
    pattern_fig.update_layout(title="Price Change % When Mars in Fire Signs")
    
    # Cycle chart
    cycle_fig = go.Figure()
    
    cycles = results['dominant_cycles']
    cycle_fig.add_trace(go.Bar(
        x=[f"Cycle {i+1}" for i in range(len(cycles))],
        y=cycles,
        name='Dominant Cycles'
    ))
    
    cycle_fig.update_layout(title="Dominant Cycle Periods (Days)")
    
    # Detected patterns text
    patterns_text = []
    
    if not results['nine_candle_patterns'].empty:
        patterns_text.append(f"Detected {len(results['nine_candle_patterns'])} instances of the 9-candle energy shift pattern")
    
    if not results['mars_fire_analysis'].empty:
        avg_change = results['mars_fire_analysis']['price_change'].mean()
        patterns_text.append(f"Mars in fire signs: Average price change of {avg_change:.2f}%")
    
    if not results['neptune_rsi_analysis'].empty:
        avg_change = results['neptune_rsi_analysis']['price_change'].mean()
        patterns_text.append(f"Neptune square Sun with RSI < 30: Average price change of {avg_change:.2f}%")
    
    patterns_html = html.Div([html.P(text) for text in patterns_text]) if patterns_text else "No significant patterns detected"
    
    return price_fig, pattern_fig, cycle_fig, patterns_html

# Callback to handle trading functionality
@app.callback(
    Output('trade-button', 'n_clicks'),
    [Input('trade-button', 'n_clicks')]
)
def start_trading(n_clicks):
    if n_clicks > 0:
        # Example trading logic
        exchange = ccxt.binance({
            'apiKey': 'YOUR_API_KEY',
            'secret': 'YOUR_SECRET_KEY',
        })
        
        symbol = 'BTC/USDT'
        order = exchange.create_market_buy_order(symbol, 0.001)
        print(f"Order executed: {order}")
    
    return 0

# Callback to trigger trading functionality immediately
@app.callback(
    Output('trigger-trade-button', 'n_clicks'),
    [Input('trigger-trade-button', 'n_clicks')]
)
def trigger_trading(n_clicks):
    global trading_triggered
    if n_clicks > 0 and not trading_triggered:
        # Example trading logic
        exchange = ccxt.binance({
            'apiKey': 'YOUR_API_KEY',
            'secret': 'YOUR_SECRET_KEY',
        })
        
        symbol = 'BTC/USDT'
        order = exchange.create_market_buy_order(symbol, 0.001)
        print(f"Order executed: {order}")
        
        # Set the flag to True to ensure trading is triggered only once
        trading_triggered = True
    
    return 0

if __name__ == '__main__':
    app.run_server(debug=True)
