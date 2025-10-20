# Get Started
## Installation
```shell
cd examples
python3 -m venv .venv
. ./.venv/bin/activate
pip install -r requirements.txt
```
## Set Up Environment Variables
### How to get secret key
Check existence after login at dashboard.eyepop.ai. If forget/not there, recreate/create the api key and store in secure place.
### Export them
```shell
export EYEPOP_SECRET_KEY=... # your secret key
```
## Run an example script
```shell
python pop_demo.py \
  --pop person \
  --output \
  --local-path ./example.jpg
```