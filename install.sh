#!/bin/bash
# install.sh - BertGCN Installation Script
# Installs the framework and dependencies.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Helper functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Poetry is installed
check_poetry() {
    if ! command -v poetry &> /dev/null; then
        print_error "Poetry is not installed"
        echo ""
        echo "Install Poetry with:"
        echo "  curl -sSL https://install.python-poetry.org | python3 -"
        echo ""
        echo "Or visit: https://python-poetry.org/docs/#installation"
        exit 1
    fi
    print_success "Poetry found: $(poetry --version)"
}

# Install framework
install_framework() {
    print_header "Installing BertGCN Framework"
    
    cd "$SCRIPT_DIR"
    
    print_info "Running: poetry install --no-interaction"
    poetry install --no-interaction
    
    print_success "Framework installed"
}

# Main installation flow
main() {
    print_header "BertGCN Installation"
    echo ""
    
    # Check prerequisites
    check_poetry
    echo ""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                echo "Usage: $0"
                echo ""
                echo "Installs the BertGCN framework."
                echo ""
                echo "After installation, use:"
                echo "  poetry run bertgcn preprocess"
                echo "  poetry run bertgcn build-graph"
                echo "  poetry run bertgcn train"
                echo "  poetry run bertgcn finetune"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Install framework
    install_framework
    echo ""
    
    # Final message
    print_header "Installation Complete!"
    echo ""
    print_success "BertGCN framework is ready to use"
    echo ""
    print_info "Quick start:"
    echo "  poetry run bertgcn train"
    echo ""
    print_info "For more commands:"
    echo "  poetry run bertgcn --help"
    echo ""
}

# Run main function
main "$@"