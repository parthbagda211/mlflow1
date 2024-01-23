venv:
	@echo "Create python environment.."
	python -m venv my-env

activate:
	@echo "Activating virtual environment..."
	env\Scripts\activate

# Target: install
# Description: Install dependencies from requirements.txt
install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt

# Target: run
# Description: Run the train.py script with specified parameters
run:
	@echo "Running train.py with parameters ephochs=$(ephochs) and lr=$(lr)..."
	@python train.py $(ephochs) $(lr)

# Target: stop
# Description: Deactivate the virtual environment
stop:
	@echo "Deactivating virtual environment..."
	deactivate

.PHONY: activate install run stop