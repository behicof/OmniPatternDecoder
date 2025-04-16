import numpy as np
import pandas as pd
import yfinance as yf
from scipy.fft import fft
from scipy.stats import spearmanr
import ephem
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class OmniPatternDecoder:
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.astro_scaler = MinMaxScaler()
        
    def fetch_market_data(self, symbol, start_date, end_date, interval='1d'):
        """Fetch market data from Yahoo Finance"""
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        return data
    
    def get_astronomical_data(self, dates):
        """Calculate astronomical positions for given dates"""
        astro_data = []
        
        for date in dates:
            # Convert to ephem date format
            ephem_date = ephem.Date(date)
            
            # Get moon phase
            moon = ephem.Moon(ephem_date)
            moon_phase = moon.phase / 100.0  # Normalize to 0-0.3 range
            
            # Get Mars position
            mars = ephem.Mars(ephem_date)
            mars.compute(ephem_date)
            # Convert to degrees in zodiac (0-360)
            mars_pos = float(mars.hlong) * 180.0 / np.pi
            
            # Get Neptune position
            neptune = ephem.Neptune(ephem_date)
            neptune.compute(ephem_date)
            neptune_pos = float(neptune.hlong) * 180.0 / np.pi
            
            # Check for fire signs positions (Aries: 0-30, Leo: 120-150, Sagittarius: 240-270)
            mars_in_fire = (0 <= mars_pos <= 30) or (120 <= mars_pos <= 150) or (240 <= mars_pos <= 270)
            
            # Check for Neptune square with Sun
            sun = ephem.Sun(ephem_date)
            sun.compute(ephem_date)
            sun_pos = float(sun.hlong) * 180.0 / np.pi
            
            # Calculate angular distance
            angle_diff = abs(neptune_pos - sun_pos)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            neptune_square_sun = 85 <= angle_diff <= 95  # Approximately 90 degrees
            
            astro_data.append([
                moon_phase,
                mars_pos,
                neptune_pos,
                1 if mars_in_fire else 0,
                1 if neptune_square_sun else 0
            ])
            
        return np.array(astro_data)
    
    def calculate_cosmic_cycle(self, price_data):
        """Find dominant cycles using FFT"""
        close_prices = price_data['Close'].values
        fft_result = np.abs(fft(close_prices))
        # Get the most dominant frequencies (ignoring the DC component)
        dominant_freqs = np.argsort(fft_result[1:])[-5:] + 1
        
        # Calculate the period of these frequencies
        periods = len(close_prices) / dominant_freqs
        
        return periods
    
    def detect_9_candle_pattern(self, price_data):
        """Detect the 9-candle energy shift pattern"""
        results = []
        rsi_period = 14
        
        # Calculate RSI
        delta = price_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate volume pressure
        volume_sma = price_data['Volume'].rolling(window=20).mean()
        volume_pressure = price_data['Volume'] / volume_sma
        
        # Look for patterns every 9 candles
        for i in range(9, len(price_data), 9):
            window = price_data.iloc[i-9:i]
            rsi_change = rsi.iloc[i-1] - rsi.iloc[i-9]
            vol_pressure = volume_pressure.iloc[i-1]
            
            if abs(rsi_change) > 10 and vol_pressure > 1.2:
                results.append({
                    'date': price_data.index[i-1],
                    'rsi_change': rsi_change,
                    'volume_pressure': vol_pressure,
                    'pattern_strength': abs(rsi_change) * vol_pressure
                })
                
        return pd.DataFrame(results)
    
    def mars_fire_sign_analysis(self, price_data, astro_data, dates):
        """Analyze price movements when Mars is in fire signs"""
        results = []
        
        for i, date in enumerate(dates):
            if i < 3 or i >= len(dates) - 3:
                continue
                
            if astro_data[i][3] == 1:  # Mars in fire sign
                price_before = price_data['Close'].iloc[i-3:i].mean()
                price_after = price_data['Close'].iloc[i:i+3].mean()
                price_change = ((price_after - price_before) / price_before) * 100
                
                results.append({
                    'date': date,
                    'price_change': price_change,
                    'mars_position': astro_data[i][1]
                })
                
        return pd.DataFrame(results)
    
    def neptune_rsi_correlation(self, price_data, astro_data, dates):
        """Analyze RSI when Neptune is square with Sun"""
        results = []
        
        # Calculate RSI
        delta = price_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        for i, date in enumerate(dates):
            if i >= len(rsi):
                continue
                
            if astro_data[i][4] == 1 and rsi.iloc[i] < 30:  # Neptune square Sun and RSI < 30
                next_5_days = price_data['Close'].iloc[i:i+5] if i+5 < len(price_data) else price_data['Close'].iloc[i:]
                price_change = ((next_5_days[-1] - price_data['Close'].iloc[i]) / price_data['Close'].iloc[i]) * 100
                
                results.append({
                    'date': date,
                    'rsi': rsi.iloc[i],
                    'price_change': price_change
                })
                
        return pd.DataFrame(results)
    
    def run_analysis(self, symbol, start_date, end_date):
        """Run the complete analysis"""
        # Fetch market data
        price_data = self.fetch_market_data(symbol, start_date, end_date)
        dates = [d.to_pydatetime() for d in price_data.index]
        
        # Get astronomical data
        astro_data = self.get_astronomical_data(dates)
        
        # Run analyses
        cycles = self.calculate_cosmic_cycle(price_data)
        nine_candle = self.detect_9_candle_pattern(price_data)
        mars_analysis = self.mars_fire_sign_analysis(price_data, astro_data, dates)
        neptune_analysis = self.neptune_rsi_correlation(price_data, astro_data, dates)
        
        return {
            'price_data': price_data,
            'astro_data': astro_data,
            'dominant_cycles': cycles,
            'nine_candle_patterns': nine_candle,
            'mars_fire_analysis': mars_analysis,
            'neptune_rsi_analysis': neptune_analysis
        }
    
    def visualize_results(self, results):
        """Create visualizations for the analysis results"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 18))
        
        # Price chart with 9-candle pattern markers
        axes[0].plot(results['price_data'].index, results['price_data']['Close'])
        
        if not results['nine_candle_patterns'].empty:
            pattern_dates = results['nine_candle_patterns']['date']
            pattern_prices = [results['price_data'].loc[date, 'Close'] for date in pattern_dates]
            axes[0].scatter(pattern_dates, pattern_prices, color='red', marker='^', s=100)
        
        axes[0].set_title('Price Chart with 9-Candle Pattern Markers')
        axes[0].set_ylabel('Price')
        
        # Mars in fire signs analysis
        if not results['mars_fire_analysis'].empty:
            axes[1].bar(
                results['mars_fire_analysis']['date'], 
                results['mars_fire_analysis']['price_change']
            )
            axes[1].set_title('Price Change % When Mars in Fire Signs')
            axes[1].set_ylabel('Price Change %')
        
        # Neptune-RSI correlation
        if not results['neptune_rsi_analysis'].empty:
            axes[2].bar(
                results['neptune_rsi_analysis']['date'], 
                results['neptune_rsi_analysis']['price_change']
            )
            axes[2].set_title('Price Change % After Neptune Square Sun with RSI < 30')
            axes[2].set_ylabel('Price Change %')
        
        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    decoder = OmniPatternDecoder()
    results = decoder.run_analysis('GC=F', '2020-01-01', '2023-01-01')  # Gold futures
    decoder.visualize_results(results)
    plt.savefig('omnipattern_analysis.png')
    plt.show()