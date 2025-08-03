# BertGCN MLOps Framework Makefile
# Production-ready automation for development, testing, and deployment

.PHONY: help install test lint format clean build deploy monitor docs

# Configuration
PYTHON_VERSION = 3.8
PROJECT_NAME = bertgcn
IMAGE_NAME = bertgcn
REGISTRY = ghcr.io
NAMESPACE = clinical-ai

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[0;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "$(BLUE)🏥 BertGCN MLOps Framework$(NC)"
	@echo "$(YELLOW)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

## Development Environment
install: ## Install dependencies with Poetry
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	poetry install --with dev,monitoring
	poetry run pre-commit install
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

install-prod: ## Install production dependencies only
	@echo "$(BLUE)📦 Installing production dependencies...$(NC)"
	poetry install --only=main
	@echo "$(GREEN)✅ Production dependencies installed$(NC)"

setup: ## Setup development environment
	@echo "$(BLUE)🔧 Setting up development environment...$(NC)"
	poetry install --with dev,monitoring
	poetry run pre-commit install
	docker-compose -f docker-compose.yml up -d postgres redis mlflow
	@echo "$(GREEN)✅ Development environment ready$(NC)"

## Code Quality
lint: ## Run linting checks
	@echo "$(BLUE)🔍 Running linting checks...$(NC)"
	poetry run black --check .
	poetry run isort --check-only .
	poetry run flake8 .
	poetry run mypy src/
	poetry run bandit -r src/
	@echo "$(GREEN)✅ Linting passed$(NC)"

format: ## Format code
	@echo "$(BLUE)🎨 Formatting code...$(NC)"
	poetry run black .
	poetry run isort .
	@echo "$(GREEN)✅ Code formatted$(NC)"

security: ## Run security checks
	@echo "$(BLUE)🔒 Running security checks...$(NC)"
	poetry run bandit -r src/
	poetry run safety check
	@echo "$(GREEN)✅ Security checks passed$(NC)"

## Testing
test: ## Run tests
	@echo "$(BLUE)🧪 Running tests...$(NC)"
	poetry run pytest tests/ -v --cov=src/bertgcn --cov-report=html --cov-report=term
	@echo "$(GREEN)✅ Tests completed$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)🧪 Running unit tests...$(NC)"
	poetry run pytest tests/unit/ -v
	@echo "$(GREEN)✅ Unit tests completed$(NC)"

test-integration: ## Run integration tests
	@echo "$(BLUE)🧪 Running integration tests...$(NC)"
	poetry run pytest tests/integration/ -v
	@echo "$(GREEN)✅ Integration tests completed$(NC)"

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)🧪 Running end-to-end tests...$(NC)"
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
	@echo "$(GREEN)✅ E2E tests completed$(NC)"

## Data Operations
data-validate: ## Validate training data
	@echo "$(BLUE)📊 Validating training data...$(NC)"
	poetry run bertgcn data validate data/clinical_data.csv
	@echo "$(GREEN)✅ Data validation completed$(NC)"

data-profile: ## Generate data profiling report
	@echo "$(BLUE)📈 Generating data profile...$(NC)"
	poetry run bertgcn data profile data/clinical_data.csv --output-path reports/data_profile.html
	@echo "$(GREEN)✅ Data profile generated$(NC)"

## Training
train: ## Train model with default configuration
	@echo "$(BLUE)🚀 Training model...$(NC)"
	poetry run bertgcn train start --doclevel letter --nepochs 50
	@echo "$(GREEN)✅ Training completed$(NC)"

train-experiment: ## Train model with experiment tracking
	@echo "$(BLUE)🚀 Starting training experiment...$(NC)"
	poetry run bertgcn train start --config configs/experiments/clinical_bert_gcn.yaml
	@echo "$(GREEN)✅ Experiment completed$(NC)"

train-hpo: ## Run hyperparameter optimization
	@echo "$(BLUE)🔍 Running hyperparameter optimization...$(NC)"
	poetry run python scripts/hyperparameter_optimization.py
	@echo "$(GREEN)✅ HPO completed$(NC)"

## Model Management
model-list: ## List available models
	@echo "$(BLUE)📋 Listing models...$(NC)"
	poetry run bertgcn model list

model-validate: ## Validate model performance
	@echo "$(BLUE)🧪 Validating model...$(NC)"
	poetry run bertgcn model validate models:/bertgcn_clinical/latest
	@echo "$(GREEN)✅ Model validation completed$(NC)"

model-promote: ## Promote model to production
	@echo "$(BLUE)⬆️  Promoting model to production...$(NC)"
	poetry run bertgcn model promote bertgcn_clinical $(VERSION) Production
	@echo "$(GREEN)✅ Model promoted$(NC)"

## Serving
serve: ## Start model serving API
	@echo "$(BLUE)🚀 Starting API server...$(NC)"
	poetry run bertgcn serve start --host 0.0.0.0 --port 8000 --workers 4

serve-dev: ## Start development server
	@echo "$(BLUE)🚀 Starting development server...$(NC)"
	poetry run uvicorn bertgcn.api.serving:app --reload --host 0.0.0.0 --port 8000

api-test: ## Test API endpoints
	@echo "$(BLUE)🧪 Testing API...$(NC)"
	poetry run bertgcn serve test --url http://localhost:8000
	@echo "$(GREEN)✅ API test completed$(NC)"

## Docker Operations
build: ## Build Docker image
	@echo "$(BLUE)🐳 Building Docker image...$(NC)"
	docker build -t $(IMAGE_NAME):latest -t $(IMAGE_NAME):$(shell git rev-parse --short HEAD) .
	@echo "$(GREEN)✅ Docker image built$(NC)"

build-prod: ## Build production Docker image
	@echo "$(BLUE)🐳 Building production Docker image...$(NC)"
	docker build --target production -t $(IMAGE_NAME):prod .
	@echo "$(GREEN)✅ Production image built$(NC)"

push: ## Push Docker image to registry
	@echo "$(BLUE)📤 Pushing Docker image...$(NC)"
	docker tag $(IMAGE_NAME):latest $(REGISTRY)/$(NAMESPACE)/$(IMAGE_NAME):latest
	docker push $(REGISTRY)/$(NAMESPACE)/$(IMAGE_NAME):latest
	@echo "$(GREEN)✅ Docker image pushed$(NC)"

## Environment Management
up: ## Start development environment
	@echo "$(BLUE)🚀 Starting development environment...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✅ Environment started$(NC)"

down: ## Stop development environment
	@echo "$(BLUE)🛑 Stopping development environment...$(NC)"
	docker-compose down
	@echo "$(GREEN)✅ Environment stopped$(NC)"

logs: ## Show container logs
	@echo "$(BLUE)📜 Showing logs...$(NC)"
	docker-compose logs -f

restart: ## Restart development environment
	@echo "$(BLUE)🔄 Restarting environment...$(NC)"
	docker-compose restart
	@echo "$(GREEN)✅ Environment restarted$(NC)"

## Monitoring
monitor: ## Start monitoring stack
	@echo "$(BLUE)📊 Starting monitoring stack...$(NC)"
	docker-compose -f docker-compose.monitoring.yml up -d
	@echo "$(GREEN)✅ Monitoring started$(NC)"
	@echo "$(YELLOW)📊 Grafana: http://localhost:3000$(NC)"
	@echo "$(YELLOW)📈 Prometheus: http://localhost:9090$(NC)"

monitor-down: ## Stop monitoring stack
	@echo "$(BLUE)🛑 Stopping monitoring stack...$(NC)"
	docker-compose -f docker-compose.monitoring.yml down
	@echo "$(GREEN)✅ Monitoring stopped$(NC)"

## Deployment
deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)🚀 Deploying to staging...$(NC)"
	kubectl apply -f k8s/staging/ -n staging
	@echo "$(GREEN)✅ Deployed to staging$(NC)"

deploy-prod: ## Deploy to production environment
	@echo "$(BLUE)🚀 Deploying to production...$(NC)"
	kubectl apply -f k8s/production/ -n production
	@echo "$(GREEN)✅ Deployed to production$(NC)"

rollback: ## Rollback deployment
	@echo "$(BLUE)🔄 Rolling back deployment...$(NC)"
	kubectl rollout undo deployment/bertgcn-api -n production
	@echo "$(GREEN)✅ Rollback completed$(NC)"

## Documentation
docs: ## Generate documentation
	@echo "$(BLUE)📚 Generating documentation...$(NC)"
	poetry run mkdocs build
	@echo "$(GREEN)✅ Documentation generated$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)📚 Serving documentation...$(NC)"
	poetry run mkdocs serve

## Utilities
clean: ## Clean temporary files and caches
	@echo "$(BLUE)🧹 Cleaning temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	docker system prune -f
	@echo "$(GREEN)✅ Cleanup completed$(NC)"

backup: ## Backup important data
	@echo "$(BLUE)💾 Creating backup...$(NC)"
	mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	docker-compose exec postgres pg_dump -U bertgcn bertgcn > backups/$(shell date +%Y%m%d_%H%M%S)/database.sql
	@echo "$(GREEN)✅ Backup created$(NC)"

check-deps: ## Check for dependency updates
	@echo "$(BLUE)🔍 Checking for dependency updates...$(NC)"
	poetry show --outdated
	@echo "$(GREEN)✅ Dependency check completed$(NC)"

## CI/CD
ci: ## Run CI pipeline locally
	@echo "$(BLUE)🔄 Running CI pipeline...$(NC)"
	make lint
	make security
	make test
	make build
	@echo "$(GREEN)✅ CI pipeline completed$(NC)"

pre-commit: ## Run pre-commit hooks
	@echo "$(BLUE)🔍 Running pre-commit hooks...$(NC)"
	poetry run pre-commit run --all-files
	@echo "$(GREEN)✅ Pre-commit checks completed$(NC)"

## Performance
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)⚡ Running benchmarks...$(NC)"
	poetry run python scripts/benchmark.py
	@echo "$(GREEN)✅ Benchmarks completed$(NC)"

profile: ## Profile model performance
	@echo "$(BLUE)📊 Profiling model performance...$(NC)"
	poetry run python scripts/profile_model.py
	@echo "$(GREEN)✅ Profiling completed$(NC)"

## Database
db-migrate: ## Run database migrations
	@echo "$(BLUE)🗃️  Running database migrations...$(NC)"
	poetry run alembic upgrade head
	@echo "$(GREEN)✅ Migrations completed$(NC)"

db-reset: ## Reset database
	@echo "$(BLUE)🗃️  Resetting database...$(NC)"
	docker-compose exec postgres psql -U bertgcn -c "DROP DATABASE IF EXISTS bertgcn; CREATE DATABASE bertgcn;"
	make db-migrate
	@echo "$(GREEN)✅ Database reset$(NC)"

## Utilities for checking status
status: ## Show system status
	@echo "$(BLUE)📊 System Status$(NC)"
	@echo "$(YELLOW)Docker Containers:$(NC)"
	@docker-compose ps
	@echo "$(YELLOW)Available Models:$(NC)"
	@poetry run bertgcn model list || echo "No models available"
	@echo "$(YELLOW)System Resources:$(NC)"
	@docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"