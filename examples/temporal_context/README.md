# Temporal context examples

Pops that run detection and tracking on every frame, and deep analysis only once per tracked object, on the frame that shows it best, through the `select_crop` forward operator. Each script is the complete code of a tutorial in the [Temporal Context in Videos](https://docs.eyepop.ai/developer-documentation/platform/temporal-context) section of the EyePop documentation.

Requires `eyepop>=3.24.0` and `EYEPOP_API_KEY` (see [../README.md](../README.md#credentials)).

| Script | What it does |
|---|---|
| `create_make_model_ability.py` | Creates and publishes the `<namespace>.describe.vehicle-make-model` VLM ability, with a JSON response schema for make, model and color. Run it once before `drive_through_make_model.py`. |
| `drive_through_make_model.py` | Names the make, model and color of the ten cars entering and leaving the drive-through in `media/drive_through.mp4`, with one VLM call per car, and tells which way each one drove. |
| `walk_by_face_id.py` | Keeps the best face-ID vector of each of the ten walk-bys in `media/office_walk_by.mp4`, and recognizes the two people who come back in different clothes. |

```shell
python create_make_model_ability.py --account <account-uuid> --namespace <namespace>
python drive_through_make_model.py --ability <namespace>.describe.vehicle-make-model:latest
python walk_by_face_id.py --output faces.jsonl
```

`eyepop get accounts` lists your account uuid; `eyepop get accounts <account-uuid>` lists its namespaces.

Every script takes `--video` to run on another file or on an `rtsp://` camera instead.

## The videos

The videos in `media/` are synthetic: generated with Google Veo from text prompts and generated start frames, so every person and car in them is generated too.
