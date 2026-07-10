#!/usr/bin/env bash
cd "$(dirname "$0")"
pip install -r requirements.txt -q
streamlit run app.py
