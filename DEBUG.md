# Debug info for discovering pipeline requests and issues

1. Here is the base cURL with a valid header
```shell
curl --location 'https://sessions.staging.eyepop.xyz/3b3a8b91-142f-4423-bb56-e772107fbaa6/pipelines' \
--header 'Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InZUdzF6bi02cjFPcXg0NmNxRl9PMiJ9.eyJodHRwczovL2lkZW50LmV5ZXBvcC5haS9lbWFpbCI6InJ5YW5AZXllcG9wLmFpIiwiaHR0cHM6Ly9pZGVudC5leWVwb3AuYWkvYXV0aC1wcm92aWRlci1pZCI6ImF1dGgwfDY1MjQ3NjMzZjE5YTRjY2NiODBkNzZjMyIsImh0dHBzOi8vY2xhaW1zLmV5ZXBvcC5haS9ncmFudHMiOlt7InBlcm1pc3Npb24iOiJhY2Nlc3M6ZGF0YXNldHMiLCJ0YXJnZXQiOiJhY2NvdW50OjAzNGNiOGUzN2Y1NDQ0ZTk4YTc4ZjFiZTY1ZmQwYmZmIn0seyJwZXJtaXNzaW9uIjoiYWNjZXNzOmRhdGFzZXRzIiwidGFyZ2V0IjoiYWNjb3VudDpkZTc0ZDM4N2I1ZjU0ZmJkYmNlMmY0ZTRkNjllMjVjZSJ9LHsicGVybWlzc2lvbiI6ImFjY2VzczpkYXRhc2V0cyIsInRhcmdldCI6ImFjY291bnQ6Yzc4YzAxMTliM2NiNDU3YTkwYTMwZjZkNDNmMmQ2ZjQifSx7InBlcm1pc3Npb24iOiJhY2Nlc3M6ZGF0YXNldHMiLCJ0YXJnZXQiOiJhY2NvdW50OmM4NTcwYzViM2YyMzQ1ZTdiMjVhZmI3NDQ5NDkxOGI3In0seyJwZXJtaXNzaW9uIjoiYWNjZXNzOmRhdGFzZXRzIiwidGFyZ2V0IjoiYWNjb3VudDoyOTgyMThiZGI1Njg0OGVmYmEwM2E1N2IxZTEzMWNmMiJ9LHsicGVybWlzc2lvbiI6ImFjY2VzczpkYXRhc2V0cyIsInRhcmdldCI6ImFjY291bnQ6NDkzMjZmMmUwODVhNDZjMzliYTczZjkxYzUyZTQzNmMifSx7InBlcm1pc3Npb24iOiJhY2Nlc3M6ZGF0YXNldHMiLCJ0YXJnZXQiOiJhY2NvdW50OmYzN2ZiMmNiY2U3NDRlODNhZGY0YzcxOGFiMDk0YTk5In0seyJwZXJtaXNzaW9uIjoiYWNjZXNzOmluZmVyZW5jZS1hcGkiLCJ0YXJnZXQiOiJ1c2VyOmF1dGgwfDY1MjQ3NjMzZjE5YTRjY2NiODBkNzZjMyJ9XSwiaXNzIjoiaHR0cHM6Ly9kZXYtZXllcG9wLnVzLmF1dGgwLmNvbS8iLCJzdWIiOiJhdXRoMHw2NTI0NzYzM2YxOWE0Y2NjYjgwZDc2YzMiLCJhdWQiOiJodHRwczovL2Rldi1hcHAuZXllcG9wLmFpIiwiaWF0IjoxNzU1NDYzMjEyLCJleHAiOjE3NTU0NzA0MTIsInNjb3BlIjoiYWNjZXNzOmRhdGFzZXRzIGFjY2VzczppbmZlcmVuY2UtYXBpIGFkbWluOmNsb3VkcyBhZG1pbjpjb21wdXRlIHVzZXI6Y29tcHV0ZSIsImF6cCI6IklVdDBwczJtYVdaVWRFbUVPUFJhUEpLdmtRVHM2WVVFIiwicGVybWlzc2lvbnMiOlsiYWNjZXNzOmRhdGFzZXRzIiwiYWNjZXNzOmluZmVyZW5jZS1hcGkiLCJhZG1pbjphcmdvY2QiLCJhZG1pbjpjbG91ZC1pbnN0YW5jZXMiLCJhZG1pbjpjbG91ZHMiLCJhZG1pbjpjb21wdXRlIiwicmVhZDptb2RlbC11cmxzIiwidXNlcjpjb21wdXRlIl19.h6WZKEPwaGp8SuFVtN_b7zD_uLj4yj0K8jBjXl_DRaCD81Ygy-x3-O7J3sa00UMyiMWfrNLu9xCbmJAMDFYb76y4FBDJSJ9I18oSp6vUEeblCfzT-ofm2TMw5PnDi9MI8CMfmJn8i71g3BNIaxHaAs8rel8pb-431bTc83TvO4Cgm5ak0fNUB7cS83sauzqtXlm_OugLNyywhMzUcbyc29U_AkUqC8Z66rdOu7yc_3-1heUP82uPp2b0dDZeF9X_rVF_0XASmZei5hG1AcHia0v4P6y08aIT08wJ42_uS4BnciNb09J8j212ByPCWq7roeqLjo1Mz3o80OEP9SxLcA' \
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
