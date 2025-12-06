import argparse
import json
import logging
import time
from typing import Sequence

import requests
from dotenv import load_dotenv
from pydantic import TypeAdapter

from eyepop import EyePopSdk
from eyepop.data.data_types import Dataset

load_dotenv()

log = logging.getLogger('eyepop.example')


parser = argparse.ArgumentParser(
                    prog='List all datasets for an account',
                    description='Authentication is provided by opening a browser session',
                    epilog='.')
parser.add_argument('-a', '--account-uuid', required=True, help="list datasets for this account uuid", default=None, type=str)
parser.add_argument('-c', '--client-id', required=False, help="Auth0 client ID", default="0H1a6pF3x2sOyKLAkwsdDaX2yEqolo53", type=str)
parser.add_argument('-d', '--domain', required=False, help="Auth0 Domain", default="auth0.eyepop.ai", type=str)
parser.add_argument('-aa', '--audience', required=False, help="Auth0 audience", default="https://api.eyepop.ai", type=str)
main_args = parser.parse_args()

def login(client_id: str, domain: str, audience: str) -> str:
    """Runs the device authorization flow and stores the user object in memory."""
    device_code_payload = {
        'client_id': client_id,
        "audience": audience,
        'scope': 'openid profile access:datasets'
    }
    device_code_url = 'https://{}/oauth/device/code'.format(domain)
    log.info("device_code_url: %s | %s", device_code_url, json.dumps(device_code_payload))
    device_code_response = requests.post(device_code_url, data=device_code_payload)

    if device_code_response.status_code != 200:
        log.error(f'Error generating the device code: {device_code_response.status_code} {device_code_response.text}')
        exit(1)

    print('Device code successful')
    device_code_data = device_code_response.json()
    print('1. On your computer or mobile device navigate to: ', device_code_data['verification_uri_complete'])
    print('2. Enter the following code: ', device_code_data['user_code'])

    # New code 👇
    token_payload = {
        'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
        'device_code': device_code_data['device_code'],
        'client_id': client_id
    }

    while True:
        print('Checking if the user completed the flow...')
        token_response = requests.post('https://{}/oauth/token'.format(domain), data=token_payload)
        token_data = token_response.json()
        if token_response.status_code == 200:
            print('Authenticated!')
            return token_data['access_token']
        elif token_data['error'] not in ('authorization_pending', 'slow_down'):
            print(token_data['error_description'])
            exit(1)
        else:
            time.sleep(device_code_data['interval'])

def main():
    access_token = login(
        client_id=main_args.client_id,
        domain=main_args.domain,
        audience=main_args.audience
    )
    with EyePopSdk.dataEndpoint(access_token=access_token) as endpoint:
        datasets = endpoint.list_datasets(account_uuid=main_args.account_uuid)
        print(TypeAdapter(Sequence[Dataset]).dump_json(datasets, indent=2))

main()
