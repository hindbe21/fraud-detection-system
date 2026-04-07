.PHONY: help install train api streamlit test docker-build docker-api docker-train clean

PYTHON = python
IMAGE_TRAIN = fraud-detection-train
IMAGE_API = fraud-detection-api

help:
	@echo ""
	@echo "  Fraud Detection System — commandes disponibles"
	@echo ""
	@echo "  Setup"
	@echo "    make install       Installer les dépendances"
	@echo ""
	@echo "  Entraînement & API"
	@echo "    make train         Lancer le pipeline d'entraînement complet"
	@echo "    make api           Démarrer l'API FastAPI (port 8000)"
	@echo "    make streamlit     Démarrer le dashboard Streamlit (port 8501)"
	@echo ""
	@echo "  Tests"
	@echo "    make test          Lancer tous les tests"
	@echo ""
	@echo "  Docker"
	@echo "    make docker-build  Builder les deux images Docker"
	@echo "    make docker-api    Lancer l'API dans Docker"
	@echo "    make docker-train  Lancer l'entraînement dans Docker"
	@echo ""
	@echo "  Nettoyage"
	@echo "    make clean         Supprimer les fichiers temporaires"
	@echo ""

install:
	pip install -r requirements.txt

train:
	$(PYTHON) -m app.pipelines.training_pipeline

api:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

streamlit:
	streamlit run streamlit_app.py

test:
	pytest tests/ -v

docker-build:
	docker build -f Dockerfile.train -t $(IMAGE_TRAIN) .
	docker build -f Dockerfile.api   -t $(IMAGE_API)   .

docker-train:
	docker run --rm -v $(PWD)/artifacts:/app/artifacts -v $(PWD)/data:/app/data $(IMAGE_TRAIN)

docker-api:
	docker run --rm -p 8000:8000 -v $(PWD)/artifacts:/app/artifacts $(IMAGE_API)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true