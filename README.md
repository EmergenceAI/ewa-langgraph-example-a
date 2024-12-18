# waa-starter-kit-code

A repo of starter code kit for web-automation agent

## Step 1: Creat a new python virtual environment (python3.11 recommended) and install all the dependencies

1. Create a new virtual environment

   ```bash
   python -m venv .venv
   ```

2. Activate the environment

    ```bash
    source .venv/bin/activate
    ```

3. Install the dependencies

    ```bash
    python -m pip install -r requirements.txt
    ```

## Step 2: Get API key

Get an API key from https://dashboard.emergence.ai, and populate it and an OpenAI key in a `.env` file:

```
OPENAI_API_KEY="<your API key here>"
EMERGENCE_API_KEY="<your API key here>"
```

## Step 3: Run the starter code

1. Run the starter code

    ```bash
        python3.11 langgraph_code.py
    ```
