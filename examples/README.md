# Get Started
## Installation
```shell
pip install eyepop
```
## Set Up Environment Variables
### How to get secret key
Check existence after login at dashboard.eyepop.ai. If forget/not there, recreate/create the api key and store in secure place.
### How to get UUID
Should show in Info tab of current pipeline at dashboard.eyepop.ai.
### Export them
```shell
export EYEPOP_SECRET_KEY=... # your secret key
export EYEPOP_POP_ID=... # your pop's uuid
export EYEPOP_URL=https://staging-api.eyepop.ai # if not specified, default to "api.eyepop.ai"
```