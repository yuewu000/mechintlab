
.PHONY: toy-superposition induction viz test lint

toy-superposition:
	python experiments/toy_superposition/run.py

induction:
	python experiments/induction_heads/run.py

viz:
	streamlit run mechintlab/viz/dashboards.py

test:
	pytest -q

lint:
	ruff check . && black --check .
