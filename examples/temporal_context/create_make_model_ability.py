"""Create the VLM ability that names a car's make, model and color.

The ability's JSON response schema makes the VLM answer with an object instead
of free text, so every selected car comes back as the same three fields.
Publishing it gives it the alias `<namespace>.describe.vehicle-make-model:latest`.

Find your account uuid and namespace with `eyepop get accounts`, then:

    python create_make_model_ability.py --account <account-uuid> --namespace <namespace>

Running it again publishes a new version under the same alias.

See https://docs.eyepop.ai/developer-documentation/platform/temporal-context/example-car-make-and-model
"""
import argparse

from openapi_pydantic import Schema

from eyepop import EyePopSdk
from eyepop.data.types.vlm import (
    InferRuntimeConfig,
    VlmAbilityCreate,
    VlmAbilityGroupCreate,
)

PROMPT = """You see one vehicle, cropped from a camera mounted above a drive-through lane, so it
may be seen from above, from the front or from behind. Identify its manufacturer (make), its
model name and its main body color. Use every clue: the badge, the grille, the lights and the
shape of the body. A pickup truck, with an open cargo bed, is never an SUV. Name the model line
as precisely as you can, the way a dealer would, for example Tacoma, Ram 1500, Civic or RAV4.
Name the most likely model even when the vehicle is seen from behind, and answer "unknown" only
when the image shows too little of the vehicle to tell."""

RESPONSE_SCHEMA = Schema.model_validate({
    "type": "object",
    "properties": {
        "make": {"type": "string", "description": "Manufacturer, e.g. Toyota"},
        "model": {"type": "string", "description": "Model name, e.g. Corolla"},
        "color": {"type": "string", "description": "Main body color, e.g. silver"},
    },
    "required": ["make", "model", "color"],
    "additionalProperties": False,
})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--account", required=True, help="your account uuid, from `eyepop get accounts`")
    parser.add_argument("--namespace", required=True, help="your namespace, from `eyepop get accounts <uuid>`")
    args = parser.parse_args()

    alias = f"{args.namespace}.describe.vehicle-make-model"
    ability = VlmAbilityCreate(
        name=alias,
        description="Names the make, model and color of one cropped vehicle",
        worker_release="qwen3-instruct",
        text_prompt=PROMPT,
        response_schema=RESPONSE_SCHEMA,
        config=InferRuntimeConfig(image_size=512, max_new_tokens=100),
        is_public=False,
    )

    with EyePopSdk.dataEndpoint(account_id=args.account) as endpoint:
        # An ability group holds the versions of one ability.
        group = next((g for g in endpoint.list_vlm_ability_groups() if g.name == alias), None)
        if group is None:
            group = endpoint.create_vlm_ability_group(VlmAbilityGroupCreate(
                name=alias,
                description=ability.description,
                default_alias_name=alias,
            ))
        created = endpoint.create_vlm_ability(create=ability, vlm_ability_group_uuid=group.uuid)
        # Publishing tags the alias with a new version and with `latest`.
        endpoint.publish_vlm_ability(vlm_ability_uuid=created.uuid, alias_name=alias)

    print(f"{alias}:latest")


if __name__ == "__main__":
    main()
