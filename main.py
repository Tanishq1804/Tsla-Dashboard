#!/usr/bin/env python3
"""
TSLA Trading Dashboard - Replit Entry Point
Advanced Trading Analysis with AI Assistant
"""

import subprocess
import sys
import os
import time
from pathlib import Path


def print_banner():
    print("=" * 60)
    print("🚀 TSLA TRADING DASHBOARD")
    print("📈 Advanced Trading Analysis with AI Assistant")
    print("🤖 Powered by TradingView Charts & Google Gemini")
    print("=" * 60)


def install_dependencies():
    print("\n🔧 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
            "--quiet"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_streamlit():
    try:
        import streamlit
        print(f"✅ Streamlit v{streamlit.__version__} ready!")
        return True
    except ImportError:
        print("❌ Streamlit not found!")
        return False


def setup_environment():
    print("\n⚙️ Setting up environment...")
    env_vars = {
        "STREAMLIT_SERVER_PORT": "8080",
        "STREAMLIT_SERVER_ADDRESS": "0.0.0.0",
        "STREAMLIT_SERVER_HEADLESS": "true",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_THEME_BASE": "dark",
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    print("✅ Environment configured for Replit!")


def run_streamlit():
    print("\n🚀 Starting TSLA Trading Dashboard...")
    print("📊 Dashboard will be available at your Replit URL")
    print("🔗 Click the web preview to access the dashboard")
    print("\n" + "=" * 60)

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8080", "--server.address=0.0.0.0",
            "--server.headless=true", "--browser.gatherUsageStats=false",
            "--theme.base=dark"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user!")
    except Exception as e:
        print(f"\n❌ Error running dashboard: {e}")


def run_health_check():
    checks = [
        ("Python version", sys.version_info
         >= (3, 8), f"Python {sys.version}"),
        ("Requirements file", Path("requirements.txt").exists(),
         "requirements.txt found"),
        ("App file", Path("app.py").exists(), "app.py found"),
    ]

    all_passed = True
    for check_name, passed, details in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}: {details}")
        if not passed:
            all_passed = False

    return all_passed


def main():
    print_banner()

    if not run_health_check():
        print("\n❌ Health checks failed! Please check your file structure.")
        return

    setup_environment()

    if not install_dependencies():
        print(
            "\n❌ Failed to install dependencies. Please check your internet connection."
        )
        return

    if not check_streamlit():
        print("\n❌ Streamlit check failed. Please restart the Repl.")
        return

    time.sleep(2)
    run_streamlit()


if __name__ == "__main__":
    main()
