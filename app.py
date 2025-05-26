import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import ast
import time
import google.generativeai as genai
from typing import List, Dict, Any
import os

# Configure page
st.set_page_config(page_title="TSLA Trading Dashboard",
                   page_icon="📈",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'step' not in st.session_state:
    st.session_state.step = 1


class TradingDashboard:

    def __init__(self):
        self.data = None
        self.processed_data = None

    def load_data_from_csv(self):
        """Load TSLA data from tsla_data.csv in the project directory"""
        try:
            # Load the CSV file from the project directory
            df = pd.read_csv('tsla_data.csv')

            # Convert all column names to lowercase for case-insensitive comparison
            df.columns = df.columns.str.lower()

            # Validate required columns
            required_columns = [
                'date', 'open', 'high', 'low', 'close', 'volume', 'direction',
                'support', 'resistance'
            ]
            missing_columns = [
                col for col in required_columns if col not in df.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in tsla_data.csv: {missing_columns}. Found columns: {list(df.columns)}"
                )

            # Ensure numeric columns are properly formatted
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows with NaN values in critical columns
            df = df.dropna(subset=numeric_cols)

            return df
        except FileNotFoundError:
            raise FileNotFoundError(
                "tsla_data.csv not found in the project directory. Please ensure the file is uploaded to Replit."
            )
        except Exception as e:
            raise Exception(f"Error loading tsla_data.csv: {str(e)}")

    def parse_price_list(self, price_str):
        """Parse price list from string format"""
        if pd.isna(price_str) or price_str == '':
            return []
        try:
            if isinstance(price_str, str):
                if price_str.startswith('[') and price_str.endswith(']'):
                    return ast.literal_eval(price_str)
                else:
                    return [
                        float(x.strip()) for x in price_str.split(',')
                        if x.strip()
                    ]
            elif isinstance(price_str, list):
                return price_str
            else:
                return [float(price_str)]
        except:
            return []


class AIAssistant:

    def __init__(self, api_key=None):
        self.api_key = api_key
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                st.success("Gemini API initialized successfully.")
            except Exception as e:
                st.error(f"Failed to initialize Gemini API: {str(e)}")
                self.model = None
        else:
            st.error(
                "No Gemini API key provided. Please set the GEMINI_API_KEY in Replit's secrets to enable chatbot functionality."
            )
            self.model = None

    def analyze_data(self, df, question):
        if not self.model:
            return "Chatbot functionality is unavailable due to missing or invalid Gemini API configuration."

        # Simplified data summary for testing
        data_summary = f"TSLA stock data has {len(df)} trading days."

        prompt = f"""
        You are a trading data analyst. Based on the following data summary, answer the question.

        Data Summary: {data_summary}

        Question: {question}

        Provide a concise answer.
        """

        try:
            st.write(f"Debug: Sending prompt to Gemini API: {prompt[:100]}..."
                     )  # Log first 100 chars of prompt
            response = self.model.generate_content(prompt)
            st.write("Debug: API call completed.")
            if response and hasattr(response, 'text'):
                st.write("Debug: Response received from Gemini API.")
                return response.text
            else:
                st.write("Debug: No response text received from Gemini API.")
                return "No response received from the Gemini API."
        except Exception as e:
            st.write(f"Debug: Error in Gemini API call: {str(e)}")
            return f"Error generating response from Gemini API: {str(e)}"

    def prepare_data_summary(self, df):
        df = df.copy()
        dates = pd.to_datetime(df['date'])

        summary = f"""
        TSLA Stock Data Analysis:
        - Total trading days: {len(df)}
        - Date range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}
        - Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}
        - Average volume: {df['volume'].mean():,.0f}

        Direction Analysis:
        - LONG signals: {len(df[df['direction'] == 'LONG'])}
        - SHORT signals: {len(df[df['direction'] == 'SHORT'])}
        - Neutral signals: {len(df[df['direction'] == 'None'])}

        Price Movement:
        - Bullish days (close > open): {len(df[df['close'] > df['open']])}
        - Bearish days (close < open): {len(df[df['close'] < df['open']])}
        - Unchanged days: {len(df[df['close'] == df['open']])}

        2023 Specific Data:
        - 2023 trading days: {len(df[dates.dt.year == 2023])}
        - 2023 bullish days: {len(df[(dates.dt.year == 2023) & (df['close'] > df['open'])])}
        - 2023 LONG signals: {len(df[(dates.dt.year == 2023) & (df['direction'] == 'LONG')])}
        """
        return summary

    def get_template_questions(self):
        return [
            "How many days in 2023 was TSLA bullish?",
            "What's the average trading volume for TSLA?",
            "How many LONG signals were generated in the dataset?",
            "What was the highest and lowest price recorded?",
            "What's the ratio of bullish to bearish days?",
            "How many trading signals were generated per month?",
            "What's the correlation between volume and price movement?",
            "During which month did TSLA show the most volatility?",
            "How often do support and resistance levels get tested?",
            "What's the success rate of LONG vs SHORT signals?",
            "What was the average daily price range in 2023?",
            "Which quarter showed the best performance for TSLA?",
            "How many times did the price break through the resistance band?",
            "What's the average volume on LONG signal days vs SHORT signal days?",
            "Which day of the week had the highest average price movement?"
        ]


def main():
    st.title("📈 TSLA Trading Dashboard")
    st.markdown("Advanced Trading Analysis with AI Assistant")

    # Initialize dashboard
    dashboard = TradingDashboard()

    # Sidebar
    with st.sidebar:
        st.header("📊 Dashboard Controls")

    # Load data from tsla_data.csv
    try:
        if st.session_state.data is None:
            st.session_state.data = dashboard.load_data_from_csv()
    except Exception as e:
        st.error(str(e))
        return

    # Main content
    df = st.session_state.data
    processed_df = df.copy()
    processed_df['date'] = pd.to_datetime(
        processed_df['date']).dt.strftime('%Y-%m-%d')

    # Create tabs
    tab1, tab2, tab3 = st.tabs(
        ["📈 Trading Chart", "🤖 AI Assistant", "📊 Data Analysis"])

    with tab1:
        st.header("TSLA Candlestick Chart with Indicators (Replay)")

        # Chart controls
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input("Start Date",
                                       value=pd.to_datetime(
                                           processed_df['date'].min()))
        with col2:
            end_date = st.date_input("End Date",
                                     value=pd.to_datetime(
                                         processed_df['date'].max()))
        with col3:
            height = st.slider("Chart Height", 400, 800, 600)

        # Filter data by date range
        date_series = pd.to_datetime(processed_df['date'])
        mask = (date_series >= pd.Timestamp(start_date)) & (
            date_series <= pd.Timestamp(end_date))
        filtered_df = processed_df.loc[mask].copy()

        if not filtered_df.empty:
            # Step control with a slider
            max_step = len(filtered_df)
            step = st.slider("Select Data Step",
                             1,
                             max_step,
                             st.session_state.step,
                             key="step_slider")
            st.session_state.step = step

            # Prepare data for JavaScript
            df_subset = filtered_df.iloc[:step].copy()

            candlestick_data = []
            markers = []
            support_lower_data = []
            support_upper_data = []
            resistance_lower_data = []
            resistance_upper_data = []

            for idx, row in df_subset.iterrows():
                # Validate OHLC data
                if pd.isna(row['open']) or pd.isna(row['high']) or pd.isna(
                        row['low']) or pd.isna(row['close']):
                    continue

                # Validate date format (must be YYYY-MM-DD)
                try:
                    pd.to_datetime(row['date'])
                    candlestick_data.append({
                        "time": row['date'],  # YYYY-MM-DD string format
                        "open": float(row['open']),
                        "high": float(row['high']),
                        "low": float(row['low']),
                        "close": float(row['close'])
                    })
                except ValueError:
                    st.warning(
                        f"Skipping row {idx}: Invalid date format '{row['date']}'. Expected YYYY-MM-DD."
                    )
                    continue

                # Markers for LONG/SHORT/None
                if row['direction'] == 'LONG':
                    markers.append({
                        "time": row['date'],
                        "position": "belowBar",
                        "color": "#00FF00",
                        "shape": "arrowUp",
                        "text": "LONG",
                        "size": 2
                    })
                elif row['direction'] == 'SHORT':
                    markers.append({
                        "time": row['date'],
                        "position": "aboveBar",
                        "color": "#FF0000",
                        "shape": "arrowDown",
                        "text": "SHORT",
                        "size": 2
                    })
                elif row['direction'] == 'None':
                    markers.append({
                        "time": row['date'],
                        "position": "inBar",
                        "color": "#FFFF00",
                        "shape": "circle",
                        "text": "NEUTRAL",
                        "size": 1
                    })

                # Support and Resistance Bands
                support = dashboard.parse_price_list(row['support'])
                resistance = dashboard.parse_price_list(row['resistance'])

                if support:
                    support_lower_data.append({
                        "time": row['date'],
                        "value": float(min(support))
                    })
                    support_upper_data.append({
                        "time": row['date'],
                        "value": float(max(support))
                    })
                if resistance:
                    resistance_lower_data.append({
                        "time":
                        row['date'],
                        "value":
                        float(min(resistance))
                    })
                    resistance_upper_data.append({
                        "time":
                        row['date'],
                        "value":
                        float(max(resistance))
                    })

            # HTML and JavaScript to render the chart with Lightweight Charts
            html_code = f"""
            <div id="chart_container" style="width: 100%; height: {height}px;"></div>
            <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
            <script>
                const chartContainer = document.getElementById('chart_container');
                const chart = LightweightCharts.createChart(chartContainer, {{
                    layout: {{
                        background: {{ type: 'solid', color: '#131722' }},
                        textColor: '#D9D9D9',
                    }},
                    grid: {{
                        vertLines: {{ color: 'rgba(42, 46, 57, 0.5)' }},
                        horzLines: {{ color: 'rgba(42, 46, 57, 0.5)' }},
                    }},
                    timeScale: {{
                        borderColor: '#2A2E39',
                        timeVisible: true,
                        secondsVisible: false,
                    }},
                    rightPriceScale: {{
                        borderColor: '#2A2E39',
                        autoScale: true,
                        mode: 1,
                        alignLabels: true,
                        scaleMargins: {{ top: 0.1, bottom: 0.1 }},
                    }},
                    crosshair: {{
                        mode: 1,
                        vertLine: {{
                            color: 'rgba(224, 227, 235, 0.6)',
                            width: 1,
                            style: 2,
                            labelBackgroundColor: '#4C525E',
                        }},
                        horzLine: {{
                            color: 'rgba(224, 227, 235, 0.6)',
                            width: 1,
                            style: 2,
                            labelBackgroundColor: '#4C525E',
                        }},
                    }},
                }});

                const candlestickSeries = chart.addCandlestickSeries({{
                    upColor: '#26A69A',
                    downColor: '#EF5350',
                    borderUpColor: '#26A69A',
                    borderDownColor: '#EF5350',
                    wickUpColor: '#26A69A',
                    wickDownColor: '#EF5350',
                    priceFormat: {{
                        type: 'price',
                        precision: 2,
                        minMove: 0.01,
                    }},
                }});
                candlestickSeries.setData({json.dumps(candlestick_data)});
                candlestickSeries.setMarkers({json.dumps(markers)});

                const supportLowerSeries = chart.addLineSeries({{
                    color: 'rgba(0, 255, 0, 0.5)',
                    lineWidth: 2,
                }});
                supportLowerSeries.setData({json.dumps(support_lower_data)});

                const supportUpperSeries = chart.addLineSeries({{
                    color: 'rgba(0, 255, 0, 0.5)',
                    lineWidth: 2,
                }});
                supportUpperSeries.setData({json.dumps(support_upper_data)});

                const resistanceLowerSeries = chart.addLineSeries({{
                    color: 'rgba(255, 0, 0, 0.5)',
                    lineWidth: 2,
                }});
                resistanceLowerSeries.setData({json.dumps(resistance_lower_data)});

                const resistanceUpperSeries = chart.addLineSeries({{
                    color: 'rgba(255, 0, 0, 0.5)',
                    lineWidth: 2,
                }});
                resistanceUpperSeries.setData({json.dumps(resistance_upper_data)});

                chart.timeScale().fitContent();
            </script>
            """
            st.components.v1.html(html_code, height=height + 50)

            # Chart legend
            st.markdown("""
            **Chart Legend:**
            - 🟢 Green Arrow (↑): LONG signal (below candle)
            - 🔴 Red Arrow (↓): SHORT signal (above candle)  
            - 🟡 Yellow Circle: Neutral/No signal
            - 🟢 Green Lines: Support levels
            - 🔴 Red Lines: Resistance levels
            """)
        else:
            st.warning("No data available for the selected date range.")

    with tab2:
        st.header("🤖 AI Trading Assistant")

        # Fetch Gemini API key from Replit secrets
        api_key = os.getenv("GEMINI_API_KEY")
        ai_assistant = AIAssistant(api_key)

        # Template questions
        st.subheader("💡 Template Questions")
        template_questions = ai_assistant.get_template_questions()

        cols = st.columns(2)
        for i, question in enumerate(template_questions):
            col = cols[i % 2]
            if col.button(question, key=f"template_{i}"):
                with st.spinner("Analyzing data..."):
                    response = ai_assistant.analyze_data(
                        processed_df, question)
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": response
                    })
                    st.rerun()

        # Custom question input
        st.subheader("❓ Ask Your Own Question")
        custom_question = st.text_input(
            "Enter your question about the TSLA data:")

        if st.button("Ask AI", disabled=not custom_question):
            with st.spinner("Analyzing data..."):
                response = ai_assistant.analyze_data(processed_df,
                                                     custom_question)
                st.session_state.chat_history.append({
                    "question": custom_question,
                    "answer": response
                })
                st.rerun()

        # Chat history
        if st.session_state.chat_history:
            st.subheader("💬 Chat History")
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q: {chat['question'][:50]}..." if len(
                        chat['question']) > 50 else f"Q: {chat['question']}"):
                    st.write("**Question:**", chat['question'])
                    st.write("**Answer:**", chat['answer'])

            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

    with tab3:
        st.header("📊 Data Analysis & Statistics")

        # Enhanced statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Trading Days", len(processed_df))
            st.metric(
                "Bullish Days",
                len(processed_df[processed_df['close'] >
                                 processed_df['open']]))

        with col2:
            st.metric("Average Volume",
                      f"{processed_df['volume'].mean():,.0f}")
            st.metric("LONG Signals",
                      len(processed_df[processed_df['direction'] == 'LONG']))

        with col3:
            st.metric(
                "Price Range",
                f"${processed_df['low'].min():.2f} - ${df['high'].max():.2f}")
            st.metric("SHORT Signals",
                      len(processed_df[processed_df['direction'] == 'SHORT']))

        with col4:
            volatility = ((processed_df['high'] - processed_df['low']) /
                          processed_df['close'] * 100).mean()
            st.metric("Avg Daily Volatility", f"{volatility:.2f}%")
            st.metric("Neutral Signals",
                      len(processed_df[processed_df['direction'] == 'None']))
            avg_price_change_short = (processed_df[
                processed_df['direction'] == 'SHORT']['close'].pct_change() *
                                      100).mean()
            st.metric("Avg % Price Change (SHORT)",
                      f"{avg_price_change_short:.2f}%")

        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(processed_df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
