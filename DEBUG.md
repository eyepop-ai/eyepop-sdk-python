# Debug info for discovering pipeline requests and issues

1. Here is the base cURL with a valid header
```shell
curl --location 'https://sessions.staging.eyepop.xyz/3b3a8b91-142f-4423-bb56-e772107fbaa6/pipelines' \
--header 'Authorization: Bearer ***REMOVED***' \
--header 'Accept: application/jsonl' \
--header 'Content-Type: application/json' \
--header 'Cookie: 3b3a8b91-142f-4423-bb56-e772107fbaa6=8089afb4113c2825; faf660b9-e07e-42ba-b9ea-c5b789dcfacf=731b3e1a4f4e6aae' \
--data '{
    "pop": {
      "components": []
    },
    "source": {
      "sourceType": "NONE"
    },
    "idleTimeoutSeconds": 60,
    "logging": ["out_meta"],
    "videoOutput": "no_output"
  }'
```

The above curl should return an object and if not we need to stop and figure out why

2. The logs for the REST api this is hitting is located in kubernetes. Here is the root kubectl command for finding logs `kubectl --context staging --namespace eyepop-sessions` The deployment is called "session-*" where the "*" is the truncated session uuid
3. Logs from the actual running pipeline can be found by ssh into the running pipeline pod and searching /var/log/eyepop/*** somewhere 
4. Use the example image from ./examples/example.jpg for inferrence testing
