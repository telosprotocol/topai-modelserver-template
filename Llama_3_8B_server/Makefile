install: ## install python package
	@echo "install python package"
	@pip install -r requirements.txt

run: ## Run the FastAPI server
	@echo "🚀 Running Ray Sevre"
	@serve run deployment.yaml

clean-build: ## clean build artifacts
	@rm -rf dist

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := install