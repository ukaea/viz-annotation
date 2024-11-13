
## Overview

Below is a high level overview of the project structure:
```
.
├── data                # Sample experimental data
├── ui                  # Code for the UI
├── README.md           # This README doc
└── requirements.txt    # Python environment requirements file.
```


## Installation

1. Install `uv`: https://docs.astral.sh/uv/getting-started/installation/

2. Create a new python environment with `uv`:
```sh
uv venv --python 3.12.6 
```

3. Install the project requirements:

```sh
uv pip install -r requirements.txt
```

## Run UI

Run the ui by running the following command:

```sh
streamlit run ui/main.py
```

The UI will be available at `http://localhost:8501/`