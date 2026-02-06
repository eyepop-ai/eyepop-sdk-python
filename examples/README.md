# Get Started
## Installation
```shell
cd examples
python3 -m venv .venv
. ./.venv/bin/activate
pip install -r requirements.txt
```
## Set Up Environment Variables
### How to get an API key
Sign in to [dashboard.eyepop.ai](https://dashboard.eyepop.ai) and create or find your API key under your profile settings.
### Export them
```shell
export EYEPOP_API_KEY=... # your API key
```
## Run an example script
```shell
python pop_demo.py \
  --pop person \
  --output \
  --local-path ./example.jpg
```