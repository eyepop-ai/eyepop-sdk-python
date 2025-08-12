import click
import os
import json
from eyepop import EyePopSdk
from eyepop.data.data_types import CreateWorkflowBody, ArgoWorkflowPhase

@click.group(name="workflow", help="Manage and inspect workflows. Use these commands to start, list, and get workflow details. Set EYEPOP_SECRET_KEY to your secret key")
def workflow():
    pass

@workflow.command()
@click.option('--template-name', required=True, help='Workflow template name')
@click.option('--body', required=False, help='Request body as a JSON string')
def start_workflow(template_name, body):
    """Start a new workflow."""
    params = CreateWorkflowBody(**json.loads(body)) if body else None
    with EyePopSdk.dataEndpoint(
        account_id=os.getenv("EYEPOP_ACCOUNT_ID"),
        secret_key=os.getenv("EYEPOP_SECRET_KEY"),
        is_async=False
    ) as data_endpoint:
        result = data_endpoint.start_workflow(
            template_name=template_name,
            workflow_create=params,
        )
        click.echo(result)

@workflow.command()
@click.option('--dataset-uuid', multiple=True, help='Filter by dataset UUID')
@click.option('--model-uuid', multiple=True, help='Filter by model UUID')
@click.option('--phase', multiple=True, type=click.Choice([p.value for p in ArgoWorkflowPhase]), help='Workflow phase')
def list_workflows(dataset_uuid, model_uuid, phase):
    """List workflows."""
    with EyePopSdk.dataEndpoint(
        account_id=os.getenv("EYEPOP_ACCOUNT_ID"),
        secret_key=os.getenv("EYEPOP_SECRET_KEY"),
        is_async=False
    ) as data_endpoint:
        result = data_endpoint.list_workflows(
            dataset_uuid=list(dataset_uuid) if dataset_uuid else None,
            model_uuid=list(model_uuid) if model_uuid else None,
            phase=[ArgoWorkflowPhase(p) for p in phase] if phase else None
        )
        for wf in result:
            click.echo(wf)

@workflow.command()
@click.argument('workflow_id')
def get_workflow(workflow_id):
    """Get workflow details by ID."""
    with EyePopSdk.dataEndpoint(
        account_id=os.getenv("EYEPOP_ACCOUNT_ID"),
        secret_key=os.getenv("EYEPOP_SECRET_KEY"),
        is_async=False
    ) as data_endpoint:
        result = data_endpoint.get_workflow(workflow_id)
        click.echo(result)

if __name__ == '__main__':
    workflow()