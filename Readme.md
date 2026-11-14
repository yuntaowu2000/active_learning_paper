## Active Learning

The models are available on [Google Drive](https://drive.google.com/drive/folders/1OtSqc7P6fwnBDbbSA7g8mPSRJAh9eb64?usp=sharing)

## Setup

The project is now configured with [poetry](https://python-poetry.org/) for dependency management and packaging. 
To install the dependencies and run the code:

```bash
poetry config virtualenvs.in-project true --local # this sets the virtual environment path to be in the local directory.
poetry shell # creates the virtual environment
poetry install --no-interaction --no-root # installs the dependencies
``` 

On Windows with CUDA support, due to pytorch indexing, basic version of torch is installed. To change to a valid cuda version, reinstall the torch library by

```bash
pip uninstall torch
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```