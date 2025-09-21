# 📈 Monte Carlo Investment Planner

A Streamlit app to simulate portfolio growth using Monte Carlo methods. Enter your initial investment, annual contributions, expected return, volatility, expenses, and inflation assumptions, then view the range of possible outcomes.

### Features
- Run thousands of Monte Carlo simulations of portfolio growth
- Inputs for initial investment, annual contributions, contribution growth, return rate, volatility, expense ratio, inflation, and contribution timing (start/end)
- Results shown in **nominal** and **inflation-adjusted (real)** dollars
- Summary tables with mean, median, 10th, and 90th percentiles by year
- Interactive Plotly charts with shaded percentile bands
- Optional overlay of individual simulation paths
- CSV downloads for summary data

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it locally

1. Clone the repo and install the requirements
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app
   ```bash
   streamlit run app.py
   ```

3. Open the provided local URL (or `localhost:8501`) in your browser to view the app.
