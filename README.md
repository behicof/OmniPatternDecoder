# OmniPattern Decoder

A sophisticated financial pattern recognition system that combines market data analysis with astronomical correlations and pattern recognition techniques.

## Overview

OmniPattern Decoder is designed to discover hidden patterns and correlations between financial markets and cosmic energies. The system integrates multiple data sources and uses advanced machine learning techniques to identify recurring patterns.

## System Architecture

1. **Data Acquisition Layer**
   - Market price data (Time, Open, High, Low, Close, Volume)
   - Time phases (minutes, hours, days)
   - Astronomical data (planetary positions, moon phases, astronomical harmonics)
   - Collective psychology (using NLP and social media data)

2. **Multi-temporal Memory Layer**
   - Multi-layer LSTM with attention-based connections
   - Short-term memory for volatility
   - Long-term memory for hidden cycle discovery

3. **Cosmic Decoder Layer**
   - Discovery of periodic patterns using Fourier Transform and Wavelet
   - Astronomical synchronicity with Spearman technique and harmonic detection
   - Discovery of harmonic signals (Synchronistic Events)

4. **Pattern Discovery Layer**
   - Finding recurring patterns in different time frames (Fractal Matching)
   - Transformer models for decoding the combination of market behavior and cosmic energy

5. **Graph Discovery Layer**
   - Converting price behavior to dynamic temporal graph
   - Discovery of hubs, central nodes, and threshold points

## Installation

```bash
# Clone the repository
git clone https://github.com/behicof/OmniPatternDecoder.git
cd OmniPatternDecoder

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for Swiss Ephemeris
pip install pyswisseph
```

## Usage

Basic example:

```python
from src.cosmic_decoder.omnipattern_decoder import OmniPatternDecoder

# Initialize the decoder
decoder = OmniPatternDecoder()

# Run analysis for Gold futures
results = decoder.run_analysis('GC=F', '2020-01-01', '2023-01-01')

# Visualize results
decoder.visualize_results(results)
```

Launch the dashboard:

```bash
python src/dashboard/cosmic_dashboard.py
```

## Pattern Discoveries

Some of the patterns discovered by the system include:

1. **9-Candle Cycle**
   - Every 9 candles shows a trackable energy change (based on RSI and volume pressure)
   - Similar to the moon cycle (9 days after Full Moon, buying pressure increases)

2. **Mars Transit through Fire Degrees**
   - During Mars transit through fire degrees 12, 24, and 0 (Aries, Leo, Sagittarius), 
     signs of rapid gold price movements have been observed

3. **Hidden Relationships between RSI and Neptune Position**
   - When RSI drops below 30 while Neptune is in a Square aspect with the Sun, 
     it has generated reliable selling signals

## Documentation

For more detailed information:
- [System Architecture](docs/architecture.md)
- [Discovered Patterns](docs/patterns.md)
- [User Guide](docs/user_guide.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.