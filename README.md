# 🏥 BertGCN: Clinical Text Classification Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue.svg)](https://python-poetry.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-ready MLOps framework** combining BERT embeddings with Graph Convolutional Networks for clinical text classification. Built with modern DevOps practices, automated CI/CD, and comprehensive monitoring.

## 🚀 Quick Start Guide

### Prerequisites

Before getting started, ensure you have these installed:

- **Python 3.8.1+** ([Download here](https://www.python.org/downloads/))
- **Poetry** ([Installation guide](https://python-poetry.org/docs/#installation))
- **Docker & Docker Compose** ([Installation guide](https://docs.docker.com/get-docker/))
- **Git**

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/clinical-ai/bertgcn.git
cd bertgcn
```

#### 2. Quick Setup (Recommended)
```bash
# This single command sets up everything you need
make setup
```

**What this does:**
- ✅ Installs all Python dependencies via Poetry
- ✅ Sets up pre-commit hooks for code quality
- ✅ Starts required Docker services (PostgreSQL, Redis, MLflow)
- ✅ Configures the development environment

#### 3. Verify Installation
```bash
# Check that everything is working
make status
```

You should see:
- Docker containers running (postgres, redis, mlflow)
- CLI available: `poetry run bertgcn --help`

#### 4. Alternative: Manual Setup

If you prefer step-by-step control:

```bash
# 1. Install dependencies
make install

# 2. Start development services
make up

# 3. Verify CLI is working
poetry run bertgcn --help
```

### 🧪 Train Your First Model

Now that everything is set up, let's train a complete pipeline:

```bash
# 1. Build the document-word graph
poetry run bertgcn build-graph --doclevel letter

# 2. Fine-tune BERT on clinical text
poetry run bertgcn finetune-bert --doclevel letter --nepochs 10 --clean

# 3. Train the hybrid BertGCN model
poetry run bertgcn train-gcn --doclevel letter --nepochs 20 --mixfactor 0.7
```

### 🔍 Verification & Next Steps

Check that your models trained successfully:

```bash
# List available models
poetry run bertgcn model list

# Check output directory structure
ls -la outputs/
```

You should see:
```
outputs/
├── data/
│   ├── datasets/          # Processed clinical datasets
│   └── graphs/            # Document-word graphs
└── models/
    ├── finetuned/         # Fine-tuned BERT models
    └── gcn/              # Trained GCN models
```

## 🛠️ Development Workflow

### Available Commands

View all available commands:
```bash
make help
```

### Common Development Tasks

```bash
# Code quality
make lint              # Check code quality
make format            # Format code automatically
make test              # Run all tests

# Development environment
make up                # Start services
make down              # Stop services  
make logs              # View container logs
make clean             # Clean up temporary files

# Model operations
make train             # Train with default settings
make serve-dev         # Start development API server
make monitor           # Start monitoring dashboard
```

### Troubleshooting

#### **Poetry Installation Issues**
```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -
source ~/.bashrc  # or restart terminal

# Verify Poetry is working
poetry --version
```

#### **Docker Issues**
```bash
# Check Docker is running
docker info

# Reset environment if needed
make down
make clean
make setup
```

#### **Python Version Issues**
```bash
# Check your Python version
python --version  # Should be 3.8.1+

# If using pyenv to manage Python versions
pyenv install 3.8.1
pyenv local 3.8.1
```

#### **Port Conflicts**
```bash
# Check what's using the ports
lsof -i :5000  # MLflow
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis

# Stop conflicting services
sudo systemctl stop postgresql  # if you have system PostgreSQL
```

#### **CLI Not Working**
```bash
# Make sure you're in the right directory
pwd  # Should end with /bertgcn

# Try running directly
poetry run python -m bertgcn --help

# If nothing works, reinstall
make clean
make install
```

### 📁 Project Structure

After successful setup:

```
bertgcn/
├── outputs/                    # Generated during training
│   ├── data/
│   │   ├── datasets/          # Processed clinical datasets
│   │   └── graphs/            # Document-word graphs
│   └── models/
│       ├── finetuned/         # Fine-tuned BERT models
│       └── gcn/               # Trained GCN models
├── src/bertgcn/              # Source code
├── tests/                    # Test files
├── docker-compose.yml        # Development services
├── Makefile                  # Automation commands
└── pyproject.toml           # Project dependencies
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file for custom configuration:

```bash
# MLflow tracking
MLFLOW_TRACKING_URI=http://localhost:5000

# Model settings
PRETRAINEDMODEL=/path/to/your/bert/model

# Database
DATABASE_URL=postgresql://bertgcn:password@localhost:5432/bertgcn
```

### Custom Data

To use your own clinical data:

1. Place your data file in the project directory
2. Update the data path in `src/bertgcn/config.py`
3. Run the pipeline as usual

## 📊 Web Interfaces

After running `make setup`, you'll have access to:

- **MLflow**: http://localhost:5000 (Experiment tracking)
- **PostgreSQL**: localhost:5432 (Database)
- **Redis**: localhost:6379 (Caching)

## 🐛 Getting Help

If you encounter issues:

1. **Check the logs**: `make logs`
2. **Verify status**: `make status`
3. **Clean and retry**: `make clean && make setup`
4. **Open an issue** with the error message and system details

## 📚 Next Steps

- [Training Guide](docs/training.md) - Detailed training options
- [API Documentation](docs/api.md) - Model serving endpoints
- [Deployment Guide](docs/deployment.md) - Production deployment
- [Contributing](CONTRIBUTING.md) - How to contribute

---

**Happy Training! 🚀**

For questions or support, please open an issue on GitHub.