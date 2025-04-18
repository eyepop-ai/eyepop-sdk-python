import argparse
import asyncio
import json
import logging

from eyepop.data.data_types import Workflow, CreateWorkflow, CreateWorkflowConfig
from eyepop.eyepopsdk import EyePopSdk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.DEBUG)

log = logging.getLogger(__name__)


parser = argparse.ArgumentParser(
                    prog='Start a workflow',
                    description='Demonstrates workflow Api',
                    epilog='.')
parser.add_argument('-a', '--account-uuid', required=True, type=str, help="account uuid")
parser.add_argument('-d', '--dataset-uuid', required=True, type=str, help="dataset uuid")
parser.add_argument('-t', '--template-name', required=True, type=str, help="template name")
parser.add_argument('-c', '--config', required=False, type=str, help="config as JSON")


args = parser.parse_args()

async def main(params):
    config = json.loads(params.config) if params.config else None
    async with EyePopSdk.dataEndpoint(is_async=True) as endpoint:
        workflow = await endpoint.start_workflow(
            account_uuid=args.account_uuid,
            template_name=args.template_name,
            workflow=CreateWorkflow(
                parameters=CreateWorkflowConfig(
                    dataset_uuid=args.dataset_uuid,
                    config=config,
                )
            )
        )
        print(f'started workflow {workflow.model_dump_json()}')

asyncio.run(main(params=args))
