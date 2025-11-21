#!/bin/bash

# Start the backend API in the background
cd /app/api_backend && python api.py &

# Start the frontend Streamlit app
streamlit run /app/app.py --server.port 8501 --server.address 0.0.0.0