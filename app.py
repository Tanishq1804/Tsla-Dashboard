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

            # Convert date to proper format
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

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
        if pd.isna(price_str) or price_str == '' or price_str is None:
            return []
        try:
            if isinstance(price_str, str):
                if price_str.startswith('[') and price_str.endswith(']'):
                    return ast.literal_eval(price_str)
                else:
                    # Try to split by comma and convert to float
                    prices = [
                        float(x.strip()) for x in price_str.split(',')
                        if x.strip()
                    ]
                    return prices if prices else []
            elif isinstance(price_str, (int, float)):
                return [float(price_str)]
            elif isinstance(price_str, list):
                return [float(x) for x in price_str if not pd.isna(x)]
            else:
                return []
        except:
            return []


class AIAssistant:

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.model = None

        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                st.success("✅ Gemini API initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize Gemini API: {str(e)}")
                self.model = None
        else:
            st.warning(
                "⚠️ No Gemini API key provided. Set GEMINI_API_KEY in environment variables to enable AI features."
            )

    def analyze_data(self, df, question):
        """Analyze data and answer questions"""
        if not self.model:
            return "❌ AI Assistant is unavailable. Please configure the Gemini API key."

        try:
            # Prepare comprehensive data summary
            data_summary = self.prepare_detailed_summary(df)

            prompt = f"""
            You are an expert financial data analyst specializing in stock market analysis. 

            Based on the following TSLA stock data summary, provide a detailed and accurate answer to the user's question.

            TSLA DATA SUMMARY:
            {data_summary}

            QUESTION: {question}

            INSTRUCTIONS:
            - Provide specific numbers and calculations when possible
            - Be concise but thorough
            - If the question requires data not available in the summary, clearly state what's missing
            - Focus on actionable insights
            - Use bullet points for multiple findings

            ANSWER:
            """

            # Generate response with error handling
            response = self.model.generate_content(prompt)

            if response and hasattr(response, 'text') and response.text:
                return response.text
            else:
                return "❌ No response generated. Please try rephrasing your question."

        except Exception as e:
            error_msg = str(e)
            if "SAFETY" in error_msg.upper():
                return "❌ Content filtered by safety guidelines. Please rephrase your question."
            elif "QUOTA" in error_msg.upper() or "LIMIT" in error_msg.upper():
                return "❌ API quota exceeded. Please try again later."
            else:
                return f"❌ Error generating response: {error_msg}"

    def prepare_detailed_summary(self, df):
        """Prepare detailed data summary for AI analysis"""
        try:
            dates = pd.to_datetime(df['date'])

            # Basic statistics
            total_days = len(df)
            bullish_days = len(df[df['close'] > df['open']])
            bearish_days = len(df[df['close'] < df['open']])

            # Price statistics
            price_stats = {
                'min_price': df[['open', 'high', 'low', 'close']].min().min(),
                'max_price': df[['open', 'high', 'low', 'close']].max().max(),
                'avg_close': df['close'].mean(),
                'price_std': df['close'].std()
            }

            # Volume statistics
            volume_stats = {
                'avg_volume': df['volume'].mean(),
                'total_volume': df['volume'].sum(),
                'volume_std': df['volume'].std()
            }

            # Direction analysis
            direction_counts = df['direction'].value_counts().to_dict()

            # Time-based analysis
            df_with_dates = df.copy()
            df_with_dates['date'] = dates
            df_with_dates['year'] = df_with_dates['date'].dt.year
            df_with_dates['month'] = df_with_dates['date'].dt.month
            df_with_dates['weekday'] = df_with_dates['date'].dt.day_name()

            # Yearly breakdown
            yearly_stats = {}
            for year in df_with_dates['year'].unique():
                year_data = df_with_dates[df_with_dates['year'] == year]
                yearly_stats[year] = {
                    'total_days':
                    len(year_data),
                    'bullish_days':
                    len(year_data[year_data['close'] > year_data['open']]),
                    'avg_close':
                    year_data['close'].mean(),
                    'long_signals':
                    len(year_data[year_data['direction'] == 'LONG']),
                    'short_signals':
                    len(year_data[year_data['direction'] == 'SHORT'])
                }

            # Volatility calculation
            df_with_dates['daily_return'] = df_with_dates['close'].pct_change()
            df_with_dates['volatility'] = (
                df_with_dates['high'] -
                df_with_dates['low']) / df_with_dates['close']

            summary = f"""
TSLA STOCK DATA ANALYSIS SUMMARY:

BASIC STATISTICS:
- Total trading days: {total_days}
- Date range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}
- Bullish days (close > open): {bullish_days} ({bullish_days/total_days*100:.1f}%)
- Bearish days (close < open): {bearish_days} ({bearish_days/total_days*100:.1f}%)

PRICE ANALYSIS:
- Price range: ${price_stats['min_price']:.2f} - ${price_stats['max_price']:.2f}
- Average closing price: ${price_stats['avg_close']:.2f}
- Price volatility (std): ${price_stats['price_std']:.2f}
- Average daily volatility: {df_with_dates['volatility'].mean()*100:.2f}%

VOLUME ANALYSIS:
- Average daily volume: {volume_stats['avg_volume']:,.0f}
- Total volume: {volume_stats['total_volume']:,.0f}
- Volume standard deviation: {volume_stats['volume_std']:,.0f}

TRADING SIGNALS:
- LONG signals: {direction_counts.get('LONG', 0)}
- SHORT signals: {direction_counts.get('SHORT', 0)}
- Neutral/No signals: {direction_counts.get('None', 0)}

YEARLY BREAKDOWN:
"""
            for year, stats in yearly_stats.items():
                summary += f"""
{year}:
  - Trading days: {stats['total_days']}
  - Bullish days: {stats['bullish_days']} ({stats['bullish_days']/stats['total_days']*100:.1f}%)
  - Average close: ${stats['avg_close']:.2f}
  - LONG signals: {stats['long_signals']}
  - SHORT signals: {stats['short_signals']}
"""

            # Monthly performance
            monthly_performance = df_with_dates.groupby('month').agg({
                'close':
                'mean',
                'volume':
                'mean',
                'volatility':
                'mean'
            }).round(2)

            summary += f"""
PERFORMANCE METRICS:
- Best performing month (avg close): Month {monthly_performance['close'].idxmax()} (${monthly_performance['close'].max():.2f})
- Highest volume month: Month {monthly_performance['volume'].idxmax()} ({monthly_performance['volume'].max():,.0f})
- Most volatile month: Month {monthly_performance['volatility'].idxmax()} ({monthly_performance['volatility'].max()*100:.2f}%)
- Average daily return: {df_with_dates['daily_return'].mean()*100:.2f}%
- Return volatility: {df_with_dates['daily_return'].std()*100:.2f}%
"""

            return summary

        except Exception as e:
            return f"Error preparing data summary: {str(e)}"

    def get_template_questions(self):
        return [
            "How many days in 2023 was TSLA bullish?",
            "What's the average trading volume for TSLA?",
            "How many LONG signals were generated in the dataset?",
            "What was the highest and lowest price recorded?",
            "What's the ratio of bullish to bearish days?",
            "What's the average daily volatility of TSLA?",
            "Which month showed the best performance?",
            "What's the success rate of trading signals?",
            "How does volume correlate with price movements?",
            "What's the average daily return percentage?",
            "Which year had the most bullish days?",
            "What's the total trading volume across all days?",
            "How many neutral trading days were there?",
            "What's the price range for each year?",
            "Which weekday typically has the highest volume?"
        ]


def main():
    st.title("📈 TSLA Trading Dashboard")
    st.markdown("Advanced Trading Analysis with AI Assistant")

    # Initialize dashboard
    dashboard = TradingDashboard()

    # Sidebar
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        st.markdown(
            "Upload your TSLA CSV file or ensure tsla_data.csv exists in the project directory."
        )

    # Load data
    try:
        if st.session_state.data is None:
            st.session_state.data = dashboard.load_data_from_csv()

        if st.session_state.data is None:
            st.stop()  # Stop execution if no data is loaded

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

    # Main content
    df = st.session_state.data
    processed_df = df.copy()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(
        ["📈 Trading Chart", "🤖 AI Assistant", "📊 Data Analysis"])

    with tab1:
        st.header("TSLA Candlestick Chart with Indicators")

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
            # Step control
            max_step = len(filtered_df)
            step = st.slider("Data Points to Show",
                             1,
                             max_step,
                             min(50, max_step),
                             key="step_slider")

            # Prepare data for chart
            df_subset = filtered_df.iloc[:step].copy()

            candlestick_data = []
            markers = []
            support_data = []
            resistance_data = []

            for idx, row in df_subset.iterrows():
                # Validate OHLC data
                if any(
                        pd.isna(val) for val in
                    [row['open'], row['high'], row['low'], row['close']]):
                    continue

                # Add candlestick data
                candlestick_data.append({
                    "time": row['date'],
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close'])
                })

                # Add trading signals as markers
                if row['direction'] == 'LONG':
                    markers.append({
                        "time": row['date'],
                        "position": "belowBar",
                        "color": "#00FF00",
                        "shape": "arrowUp",
                        "text": "LONG"
                    })
                elif row['direction'] == 'SHORT':
                    markers.append({
                        "time": row['date'],
                        "position": "aboveBar",
                        "color": "#FF0000",
                        "shape": "arrowDown",
                        "text": "SHORT"
                    })

                # Add support/resistance levels
                support_prices = dashboard.parse_price_list(row['support'])
                resistance_prices = dashboard.parse_price_list(
                    row['resistance'])

                for price in support_prices:
                    support_data.append({
                        "time": row['date'],
                        "value": float(price)
                    })

                for price in resistance_prices:
                    resistance_data.append({
                        "time": row['date'],
                        "value": float(price)
                    })

            # Create the chart HTML
            chart_html = f"""
            <div id="chart_container" style="width: 100%; height: {height}px;"></div>
            <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
            <script>
                const chartContainer = document.getElementById('chart_container');
                const chart = LightweightCharts.createChart(chartContainer, {{
                    layout: {{
                        background: {{ type: 'solid', color: '#1e1e1e' }},
                        textColor: '#DDD',
                    }},
                    grid: {{
                        vertLines: {{ color: 'rgba(42, 46, 57, 0.5)' }},
                        horzLines: {{ color: 'rgba(42, 46, 57, 0.5)' }},
                    }},
                    timeScale: {{
                        borderColor: '#485c7b',
                        timeVisible: true,
                    }},
                    rightPriceScale: {{
                        borderColor: '#485c7b',
                    }},
                }});

                const candlestickSeries = chart.addCandlestickSeries({{
                    upColor: '#26a69a',
                    downColor: '#ef5350',
                    borderUpColor: '#26a69a',
                    borderDownColor: '#ef5350',
                    wickUpColor: '#26a69a',
                    wickDownColor: '#ef5350',
                }});

                candlestickSeries.setData({json.dumps(candlestick_data)});

                if ({json.dumps(markers)}.length > 0) {{
                    candlestickSeries.setMarkers({json.dumps(markers)});
                }}

                if ({json.dumps(support_data)}.length > 0) {{
                    const supportSeries = chart.addLineSeries({{
                        color: 'rgba(0, 255, 0, 0.8)',
                        lineWidth: 2,
                        title: 'Support'
                    }});
                    supportSeries.setData({json.dumps(support_data)});
                }}

                if ({json.dumps(resistance_data)}.length > 0) {{
                    const resistanceSeries = chart.addLineSeries({{
                        color: 'rgba(255, 0, 0, 0.8)',
                        lineWidth: 2,
                        title: 'Resistance'
                    }});
                    resistanceSeries.setData({json.dumps(resistance_data)});
                }}

                chart.timeScale().fitContent();
            </script>
            """

            st.components.v1.html(chart_html, height=height + 50)

            # Chart legend
            st.markdown("""
            **📊 Chart Legend:**
            - 🟢 **Green Candles**: Bullish (Close > Open)
            - 🔴 **Red Candles**: Bearish (Close < Open)
            - ⬆️ **Green Arrow**: LONG signal
            - ⬇️ **Red Arrow**: SHORT signal
            - 🟢 **Green Line**: Support levels
            - 🔴 **Red Line**: Resistance levels
            """)

            # Display current data point info
            if step <= len(df_subset):
                current_row = df_subset.iloc[step - 1]
                st.info(f"""
                **Current Data Point ({step}/{max_step}):**
                📅 Date: {current_row['date']} | 
                💰 Close: ${current_row['close']:.2f} | 
                📊 Volume: {current_row['volume']:,.0f} | 
                🎯 Signal: {current_row['direction']}
                """)
        else:
            st.warning("No data available for the selected date range.")

    with tab2:
        st.header("🤖 AI Trading Assistant")

        # Get API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            api_key = st.text_input(
                "Enter Gemini API Key (optional)",
                type="password",
                help="Get your API key from Google AI Studio")

        ai_assistant = AIAssistant(api_key)

        # Template questions
        st.subheader("💡 Quick Questions")
        template_questions = ai_assistant.get_template_questions()

        # Display template questions in columns
        cols = st.columns(3)
        for i, question in enumerate(template_questions):
            col = cols[i % 3]
            if col.button(question,
                          key=f"template_{i}",
                          use_container_width=True):
                with st.spinner("🤖 Analyzing data..."):
                    response = ai_assistant.analyze_data(
                        processed_df, question)
                    st.session_state.chat_history.append({
                        "question":
                        question,
                        "answer":
                        response,
                        "timestamp":
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.rerun()

        # Custom question input
        st.subheader("❓ Ask Your Own Question")
        custom_question = st.text_area(
            "Enter your question about the TSLA data:",
            placeholder="e.g., What was the average price in Q2 2023?",
            height=100)

        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🚀 Ask AI",
                                   disabled=not custom_question.strip())

        if ask_button and custom_question.strip():
            with st.spinner("🤖 Generating response..."):
                response = ai_assistant.analyze_data(processed_df,
                                                     custom_question)
                st.session_state.chat_history.append({
                    "question":
                    custom_question,
                    "answer":
                    response,
                    "timestamp":
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.rerun()

        # Chat history
        if st.session_state.chat_history:
            st.subheader("💬 Chat History")

            # Clear history button
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

            # Display chat history (most recent first)
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"💭 {chat['question'][:60]}..." if len(
                        chat['question']) > 60 else f"💭 {chat['question']}",
                                 expanded=(i == 0)):  # Expand most recent
                    st.markdown(
                        f"**🕒 Asked:** {chat.get('timestamp', 'Unknown time')}"
                    )
                    st.markdown(f"**❓ Question:** {chat['question']}")
                    st.markdown(f"**🤖 Answer:** {chat['answer']}")

    with tab3:
        st.header("📊 Data Analysis & Statistics")

        if processed_df is not None and len(processed_df) > 0:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            bullish_days = len(
                processed_df[processed_df['close'] > processed_df['open']])
            bearish_days = len(
                processed_df[processed_df['close'] < processed_df['open']])

            with col1:
                st.metric("📈 Total Trading Days", len(processed_df))
                st.metric("🟢 Bullish Days", bullish_days)

            with col2:
                st.metric("📊 Average Volume",
                          f"{processed_df['volume'].mean():,.0f}")
                st.metric(
                    "🎯 LONG Signals",
                    len(processed_df[processed_df['direction'] == 'LONG']))

            with col3:
                price_min = processed_df[['open', 'high', 'low',
                                          'close']].min().min()
                price_max = processed_df[['open', 'high', 'low',
                                          'close']].max().max()
                st.metric("💰 Price Range",
                          f"${price_min:.2f} - ${price_max:.2f}")
                st.metric(
                    "🎯 SHORT Signals",
                    len(processed_df[processed_df['direction'] == 'SHORT']))

            with col4:
                volatility = ((processed_df['high'] - processed_df['low']) /
                              processed_df['close'] * 100).mean()
                st.metric("📈 Avg Daily Volatility", f"{volatility:.2f}%")
                st.metric(
                    "⚪ Neutral Signals",
                    len(processed_df[processed_df['direction'] == 'None']))

            # Performance metrics
            st.subheader("📈 Performance Metrics")
            col1, col2, col3 = st.columns(3)

            with col1:
                bullish_ratio = bullish_days / len(processed_df) * 100
                st.metric("🟢 Bullish Days %", f"{bullish_ratio:.1f}%")

            with col2:
                avg_close = processed_df['close'].mean()
                st.metric("💰 Average Close Price", f"${avg_close:.2f}")

            with col3:
                total_volume = processed_df['volume'].sum()
                st.metric("📊 Total Volume", f"{total_volume:,.0f}")

            # Data preview
            st.subheader("📋 Raw Data Preview")

            # Add data filtering options
            col1, col2 = st.columns(2)
            with col1:
                show_rows = st.selectbox("Rows to display", [10, 25, 50, 100],
                                         index=0)
            with col2:
                filter_direction = st.selectbox(
                    "Filter by Direction",
                    ['All'] + list(processed_df['direction'].unique()))

            # Apply filters
            display_df = processed_df.copy()
            if filter_direction != 'All':
                display_df = display_df[display_df['direction'] ==
                                        filter_direction]

            # Display data
            st.dataframe(display_df.head(show_rows), use_container_width=True)

            # Download option
            csv = processed_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Dataset as CSV",
                data=csv,
                file_name=
                f"tsla_data_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv")
        else:
            st.error("No data available for analysis.")


if __name__ == "__main__":
    main()
