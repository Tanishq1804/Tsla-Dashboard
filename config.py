# Configuration file for TSLA Trading Dashboard

# Chart Configuration
CHART_CONFIG = {
    "theme": {
        "background_color": "#1e1e1e",
        "text_color": "white",
        "grid_color": "#2B2B43",
        "border_color": "#485c7b"
    },
    "candlestick": {
        "up_color": "#00ff88",
        "down_color": "#ff4976",
        "border_up_color": "#00ff88",
        "border_down_color": "#ff4976",
        "wick_up_color": "#00ff88",
        "wick_down_color": "#ff4976"
    },
    "signals": {
        "long_color": "#00ff00",
        "short_color": "#ff0000",
        "neutral_color": "#ffff00"
    },
    "bands": {
        "support_color": "rgba(0, 255, 0, 0.2)",
        "resistance_color": "rgba(255, 0, 0, 0.2)"
    }
}

# AI Configuration
AI_CONFIG = {
    "model_name":
    "gemini-pro",
    "max_tokens":
    1000,
    "temperature":
    0.1,
    "system_prompt":
    """You are an expert trading analyst specializing in TSLA stock analysis. 
    Provide accurate, data-driven insights based on the provided trading data. 
    Use specific numbers and percentages in your responses. 
    Be concise but comprehensive in your analysis."""
}

# Data Processing Configuration
DATA_CONFIG = {
    "required_columns": [
        "date", "open", "high", "low", "close", "volume", "direction",
        "support", "resistance"
    ],
    "date_format":
    "%Y-%m-%d",
    "price_precision":
    2,
    "volume_precision":
    0
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    "page_title": "TSLA Trading Dashboard",
    "page_icon": "📈",
    "layout": "wide",
    "default_chart_height": 600,
    "max_chart_height": 800,
    "min_chart_height": 400
}

# Sample Data Configuration
SAMPLE_DATA_CONFIG = {
    "start_date":
    "2023-01-01",
    "end_date":
    "2024-03-01",
    "base_price":
    180.0,
    "default_volume":
    25000000,
    "volatility_range": (0.015, 0.035),
    "trend_phases": [{
        "name": "bullish",
        "duration": 60,
        "bias": 0.001,
        "volatility": 0.02
    }, {
        "name": "bearish",
        "duration": 60,
        "bias": -0.001,
        "volatility": 0.025
    }, {
        "name": "sideways",
        "duration": 60,
        "bias": 0,
        "volatility": 0.015
    }, {
        "name": "volatile",
        "duration": 60,
        "bias": 0,
        "volatility": 0.035
    }]
}

# Template Questions for AI Assistant
TEMPLATE_QUESTIONS = [
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
    "What was the average daily price range?",
    "Which quarter showed the best performance?",
    "How many times did price break through resistance?",
    "What's the typical volume on signal days vs non-signal days?",
    "Show me the distribution of price movements by day of week"
]

# Error Messages
ERROR_MESSAGES = {
    "file_upload":
    "Error loading file. Please check the file format and try again.",
    "api_key": "Invalid API key. Please check your Gemini API key.",
    "data_format":
    "Data format error. Please ensure all required columns are present.",
    "date_parse": "Date parsing error. Please use YYYY-MM-DD format.",
    "price_data": "Price data error. Please ensure OHLC values are numeric.",
    "ai_request": "AI request failed. Please check your API key and try again."
}

# Success Messages
SUCCESS_MESSAGES = {
    "data_loaded": "Data loaded successfully! 🎉",
    "sample_generated": "Sample data generated successfully! 📊",
    "chart_updated": "Chart updated successfully! 📈",
    "ai_response": "Analysis complete! 🤖"
}

# Styling Constants
STYLING = {
    "sidebar_width": 300,
    "chart_container_height": 650,
    "metric_card_height": 100,
    "color_scheme": {
        "primary": "#00ff88",
        "secondary": "#ff4976",
        "accent": "#ffff00",
        "background": "#1e1e1e",
        "surface": "#2B2B43",
        "text": "#ffffff"
    }
}
