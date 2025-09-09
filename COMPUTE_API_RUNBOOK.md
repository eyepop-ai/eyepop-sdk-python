## Using compute api with the python sdk

### 1. Set your env vars for use with compute api
Setting the `EYEPOP_URL` to `https://compute.staging.eyepop.xyz` enables using the compute api. When that is set you will need a compute specific token. One can be created by an admin with scoped permissions and expiry

### 2. Use the python sdk normally
The changes here are meant to be backward compatible. The act of setting the compute url is the core switch between v1/v2 operating mode.

### 3. Here are some endpoints for working with the compute api directly:

```shell
# Create or get a session with defaults. 
# Many configuration options are avail and setting them depends on the scope of your api key. See postman for better documentation

curl --location --request POST 'https://compute.staging.eyepop.xyz/v1/sessions' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'Authorization: Bearer $EYEPOP_SECRET_KEY'


# Get a sessions info directly:
curl --location 'https://compute.staging.eyepop.xyz/v1/sessions/${SESSION_UUID}' \
--header 'Accept: application/json' \
--header 'Authorization: Bearer $EYEPOP_SECRET_KEY'

# delete a session works the same as a GET (all endpoints are RESTful)
```

