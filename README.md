# 📈 Monte Carlo Investment Planner

A Streamlit app to simulate portfolio growth using Monte Carlo methods. Enter your initial investment, annual contributions, expected return, volatility, expenses, and inflation assumptions, then view the range of possible outcomes.

### Features
- Run thousands of Monte Carlo simulations of portfolio growth
- Inputs for initial investment, annual contributions, contribution growth, return rate, volatility, expense ratio, inflation, and contribution timing (start/end)
- **Quick presets** for Conservative (5%/10%), Moderate (7%/15%), and Aggressive (9%/20%) portfolios
- **Fat-tail distribution options**: Normal, Student's t-distribution, or mixture model with configurable crash probability
- **Goal tracking**: Set a target portfolio value and see your probability of reaching it
- **Key metrics dashboard**: Median final value, 10th/90th percentiles, and goal probability at a glance
- Results shown in **nominal** and **inflation-adjusted (real)** dollars
- Summary tables with mean, median, 10th, and 90th percentiles by year
- Interactive Plotly charts with shaded percentile bands
- Optional overlay of individual simulation paths
- CSV downloads for summary data

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://investment-simulations.streamlit.app/)

### How to run it locally

1. Clone the repo and install the requirements
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app
   ```bash
   streamlit run streamlit_app.py
   ```

3. Open the provided local URL (or `localhost:8501`) in your browser to view the app.
