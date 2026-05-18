# Railway: each line below is a separate service.
# In the Railway dashboard, override the start command per service.

api:     uvicorn api.main:app --host 0.0.0.0 --port $PORT
web:     streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
worker:  python scheduler.py
