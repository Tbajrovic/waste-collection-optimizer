PY=python

setup:
	$(PY) -m venv .venv && . .venv/Scripts/activate && pip install -U pip -r requirements.txt

data:
	$(PY) Dataset/synthetic/generate_synthetic.py --days 120 --bins 500

train:
	$(PY) Code/forecasting/train.py

route:
	$(PY) Code/routing/solve_cvrp.py

demo:
	streamlit run Code/app/app.py
