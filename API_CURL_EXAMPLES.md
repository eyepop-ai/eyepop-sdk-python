# EyePop SDK API cURL Examples

This document shows the HTTP API calls that the `examples/pop_demo.py` script generates, translated into cURL commands.

## Environment Variables

```bash
export EYEPOP_URL="https://compute.staging.eyepop.xyz"
export EYEPOP_SECRET_KEY="your_secret_key_here"
export EYEPOP_USER_UUID=""  # Can be empty string
export EYEPOP_DATA_API="https://dataset-api.staging.eyepop.xyz"
```

## 1. Create Compute Session

```bash
curl -X POST "${EYEPOP_URL}/v1/session" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "user_uuid": "'${EYEPOP_USER_UUID}'"
  }'
```

Response:
```json
{
  "session_endpoint": "https://worker-xyz.eyepop.ai",
  "session_uuid": "abc123...",
  "pipeline_uuid": "def456..."
}
```

## 2. Create Pipeline

**IMPORTANT**: Due to authorization configuration on staging, you must create an empty pipeline first, then update it with inference components.

```bash
SESSION_ENDPOINT="https://worker-xyz.eyepop.ai"  # from session response

# Create empty pipeline first
curl -X POST "${SESSION_ENDPOINT}/pipelines" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
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

Response contains pipeline `id`.

**Note**: Creating pipelines with inference components directly fails with "Missing Authorization header" error due to additional user authorization checks required for inference operations.

### Available Component Types

The `components` array accepts the following component types:

#### Inference Component
```json
{
  "type": "inference",
  "id": 1,
  "inferenceTypes": ["object_detection", "key_points", "ocr", "mesh", "feature_vector", "semantic_segmentation", "segmentation", "image_classification", "raw"],
  "modelUuid": "model-uuid-string",
  "model": "eyepop.person:latest",
  "abilityUuid": "ability-uuid-string", 
  "ability": "eyepop.localize-objects:latest",
  "categoryName": "person",
  "confidenceThreshold": 0.8,
  "topK": 10,
  "targetFps": "15",
  "hidden": false,
  "params": {
    "prompts": [{"prompt": "person"}],
    "custom_param": "value"
  },
  "forward": {
    "operator": {
      "type": "crop|full|crop_with_full_fallback",
      "includeClasses": ["person", "car"],
      "crop": {
        "maxItems": 128,
        "boxPadding": 0.5,
        "orientationTargetAngle": -90.0
      }
    },
    "targets": []
  }
}
```

#### Tracing Component  
```json
{
  "type": "tracing",
  "id": 2,
  "reidModelUuid": "reid-model-uuid",
  "reidModel": "reid-model-alias",
  "maxAgeSeconds": 30.0,
  "iouThreshold": 0.3,
  "simThreshold": 0.5
}
```

#### Contour Finder Component
```json
{
  "type": "contour_finder", 
  "id": 3,
  "contourType": "polygon|all_pixels|convex_hull|hough_circles|circle|triangle|rectangle",
  "areaThreshold": 0.005
}
```

#### Component Finder Component
```json
{
  "type": "component_finder",
  "id": 4,
  "dilate": 1.0,
  "erode": 1.0,
  "keepSource": true,
  "componentClassLabel": "component"
}
```

#### Forward Component
```json
{
  "type": "forward",
  "id": 5,
  "forward": {
    "operator": {
      "type": "full"
    },
    "targets": []
  }
}
```

## 3. Update Pipeline with Pop Configuration

```bash
PIPELINE_ID="pipeline123..."  # from pipeline creation response

curl -X PATCH "${SESSION_ENDPOINT}/pipelines/${PIPELINE_ID}/pop" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "components": [
      {
        "model": "eyepop.person:latest",
        "categoryName": "person"
      }
    ]
  }'
```

## 4. Upload File for Inference

### Image Upload (--local-path image.jpg)

```bash
curl -X POST "${SESSION_ENDPOINT}/pipelines/${PIPELINE_ID}/source?mode=queue&processing=sync" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Accept: application/jsonl" \
  -F "file=@image.jpg"
```

### Image Upload with Parameters

```bash
curl -X POST "${SESSION_ENDPOINT}/pipelines/${PIPELINE_ID}/source?mode=queue&processing=sync" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Accept: application/jsonl" \
  -F "file=@image.jpg" \
  -F 'params=[{"componentId": 1, "values": {"prompts": [{"prompt": "person"}]}}]'
```

### Video Upload (Full-duplex HTTP)

Step 1 - Prepare source:
```bash
curl -X POST "${SESSION_ENDPOINT}/pipelines/${PIPELINE_ID}/prepareSource?timeout=120s" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Accept: application/jsonl"
```

Step 2 - Upload video with returned source ID:
```bash
SOURCE_ID="source123..."  # from prepare response

curl -X POST "${SESSION_ENDPOINT}/pipelines/${PIPELINE_ID}/source?mode=queue&processing=async&sourceId=${SOURCE_ID}&videoMode=exhaustive" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Accept: application/jsonl" \
  -H "Content-Type: video/mp4" \
  --data-binary "@video.mp4"
```

## 5. Load from URL (--url)

```bash
curl -X PATCH "${SESSION_ENDPOINT}/pipelines/${PIPELINE_ID}/source?mode=queue&processing=sync" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Content-Type: application/json" \
  -H "Accept: application/jsonl" \
  -d '{
    "sourceType": "URL",
    "url": "https://example.com/image.jpg"
  }'
```

### With Parameters

```bash
curl -X PATCH "${SESSION_ENDPOINT}/pipelines/${PIPELINE_ID}/source?mode=queue&processing=sync" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Content-Type: application/json" \
  -H "Accept: application/jsonl" \
  -d '{
    "sourceType": "URL",
    "url": "https://example.com/image.jpg",
    "params": [
      {
        "componentId": 1,
        "values": {
          "prompts": [{"prompt": "person"}]
        }
      }
    ]
  }'
```

## 6. Load from Asset UUID (--asset-uuid)

```bash
curl -X PATCH "${SESSION_ENDPOINT}/pipelines/${PIPELINE_ID}/source?mode=queue&processing=sync" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Content-Type: application/json" \
  -H "Accept: application/jsonl" \
  -d '{
    "sourceType": "ASSET_UUID",
    "assetUuid": "12345678-1234-1234-1234-123456789abc"
  }'
```

## 7. Complete Pipeline Examples

### Person Detection with Tracking
```bash
curl -X POST "${SESSION_ENDPOINT}/pipelines" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "pop": {
      "components": [{
        "type": "inference",
        "inferenceTypes": ["object_detection"],
        "modelUuid": "eyepop-person_EPPersonB1_Person_TorchScriptCuda_float32",
        "categoryName": "person",
        "confidenceThreshold": 0.8,
        "forward": {
          "operator": {
            "includeClasses": ["person"]
          },
          "targets": [{
            "type": "tracing",
            "reidModelUuid": "legacy:reid-mobilenetv2_x1_4_ImageNet_TensorFlowLite_int8"
          }]
        }
      }]
    },
    "source": {
      "sourceType": "NONE"
    },
    "idleTimeoutSeconds": 60,
    "logging": ["out_meta"],
    "videoOutput": "no_output"
  }'
```

### Object Detection with SAM Segmentation
```bash
curl -X POST "${SESSION_ENDPOINT}/pipelines" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "pop": {
      "components": [{
        "type": "inference",
        "inferenceTypes": ["object_detection"],
        "modelUuid": "yolov7_YOLOv7-TINY_COCO_TensorFlowLite_float32",
        "forward": {
          "operator": {
            "type": "crop",
            "crop": {
              "boxPadding": 0.5
            }
          },
          "targets": [{
            "type": "inference",
            "inferenceTypes": ["semantic_segmentation"],
            "modelUuid": "legacy:EfficientSAM_Sm_Grounded_TorchScript_float32",
            "hidden": true,
            "forward": {
              "operator": {
                "type": "full"
              },
              "targets": [{
                "type": "contour_finder",
                "contourType": "polygon",
                "areaThreshold": 0.001
              }]
            }
          }]
        }
      }]
    },
    "source": {
      "sourceType": "NONE"
    },
    "idleTimeoutSeconds": 60,
    "logging": ["out_meta"],
    "videoOutput": "no_output"
  }'
```

## 8. Pop Configuration Examples from pop_demo.py

### Person Detection (--pop person)
```json
{
  "components": [
    {
      "model": "eyepop.person:latest",
      "categoryName": "person"
    }
  ]
}
```

### 2D Body Points (--pop 2d-body-points)
```json
{
  "components": [
    {
      "model": "eyepop.person:latest",
      "categoryName": "person",
      "forward": {
        "maxItems": 128,
        "targets": [
          {
            "model": "eyepop.person.2d-body-points:latest",
            "categoryName": "2d-body-points",
            "confidenceThreshold": 0.25
          }
        ]
      }
    }
  ]
}
```

### Face Detection with Mesh (--pop faces)
```json
{
  "components": [
    {
      "model": "eyepop.person:latest",
      "categoryName": "person",
      "forward": {
        "maxItems": 128,
        "targets": [
          {
            "model": "eyepop.person.face.short-range:latest",
            "categoryName": "2d-face-points",
            "forward": {
              "boxPadding": 1.5,
              "orientationTargetAngle": -90.0,
              "targets": [
                {
                  "model": "eyepop.person.face-mesh:latest",
                  "categoryName": "3d-face-mesh"
                }
              ]
            }
          }
        ]
      }
    }
  ]
}
```

### Hand Detection (--pop hands)
```json
{
  "components": [
    {
      "model": "eyepop.person:latest",
      "categoryName": "person",
      "forward": {
        "maxItems": 128,
        "boxPadding": 0.25,
        "targets": [
          {
            "model": "eyepop.person.palm:latest",
            "forward": {
              "includeClasses": ["hand circumference"],
              "orientationTargetAngle": -90.0,
              "targets": [
                {
                  "model": "eyepop.person.3d-hand-points:latest",
                  "categoryName": "3d-hand-points"
                }
              ]
            }
          }
        ]
      }
    }
  ]
}
```

### 3D Body Points (--pop 3d-body-points)
```json
{
  "components": [
    {
      "model": "eyepop.person:latest",
      "categoryName": "person",
      "forward": {
        "boxPadding": 0.5,
        "targets": [
          {
            "model": "eyepop.person.pose:latest",
            "hidden": true,
            "forward": {
              "boxPadding": 0.5,
              "orientationTargetAngle": -90.0,
              "targets": [
                {
                  "model": "eyepop.person.3d-body-points.heavy:latest",
                  "categoryName": "3d-body-points",
                  "confidenceThreshold": 0.25
                }
              ]
            }
          }
        ]
      }
    }
  ]
}
```

### Text Recognition (--pop text)
```json
{
  "components": [
    {
      "model": "eyepop.text:latest",
      "categoryName": "text",
      "confidenceThreshold": 0.7,
      "forward": {
        "maxItems": 128,
        "targets": [
          {
            "model": "eyepop.text.recognize.landscape:latest",
            "confidenceThreshold": 0.1
          }
        ]
      }
    }
  ]
}
```

### SAM1 Segmentation (--pop sam1)
```json
{
  "components": [
    {
      "model": "eyepop.sam.small:latest",
      "id": 1,
      "forward": {
        "targets": [
          {
            "contourType": "POLYGON",
            "areaThreshold": 0.005
          }
        ]
      }
    }
  ]
}
```

### SAM2 Segmentation (--pop sam2)
```json
{
  "components": [
    {
      "model": "eyepop.sam2.encoder.tiny:latest",
      "id": 1,
      "hidden": true,
      "forward": {
        "targets": [
          {
            "model": "eyepop.sam2.decoder:latest",
            "forward": {
              "targets": [
                {
                  "contourType": "POLYGON",
                  "areaThreshold": 0.005
                }
              ]
            }
          }
        ]
      }
    }
  ]
}
```

### Image Contents Analysis (--pop image-contents)
```json
{
  "components": [
    {
      "id": 1,
      "ability": "eyepop.image-contents:latest"
    }
  ]
}
```

### Object Localization (--pop localize-objects)
```json
{
  "components": [
    {
      "id": 1,
      "ability": "eyepop.localize-objects:latest"
    }
  ]
}
```

### Object Localization with Image Contents (--pop localize-objects-plus)
```json
{
  "components": [
    {
      "id": 1,
      "ability": "eyepop.localize-objects:latest",
      "params": {
        "prompts": [{"prompt": "person"}]
      },
      "forward": {
        "targets": [
          {
            "model": "eyepop.image-contents:latest",
            "params": {
              "prompts": [
                {"prompt": "hair color blond"},
                {"prompt": "hair color brown"}
              ]
            }
          }
        ]
      }
    }
  ]
}
```

## 8. Parameter Examples

### Points/ROI (--points)
```json
{
  "componentId": 1,
  "values": {
    "roi": {
      "points": [
        {"x": 100, "y": 50},
        {"x": 200, "y": 150}
      ]
    }
  }
}
```

### Bounding Boxes (--boxes)
```json
{
  "componentId": 1,
  "values": {
    "roi": {
      "boxes": [
        {
          "topLeft": {"x": 100, "y": 50},
          "bottomRight": {"x": 200, "y": 150}
        }
      ]
    }
  }
}
```

### Prompts (--prompt)
```json
{
  "componentId": 1,
  "values": {
    "prompts": [
      {"prompt": "person"},
      {"prompt": "car"}
    ]
  }
}
```

### Single Prompt (--single-prompt)
```json
{
  "componentId": 1,
  "values": {
    "prompt": "person"
  }
}
```

## 9. Model UUID Examples

### Using Model UUID (--model-uuid)
```json
{
  "components": [
    {
      "id": 1,
      "abilityUuid": "12345678-1234-1234-1234-123456789abc"
    }
  ]
}
```

### Using Model Alias (--model-alias)
```json
{
  "components": [
    {
      "id": 1,
      "ability": "eyepop.person:latest"
    }
  ]
}
```

## 10. Response Format

Responses are in JSONL format (one JSON object per line):

```json
{"prediction": {"objects": [{"classId": 0, "className": "person", "confidence": 0.95, "x": 100, "y": 50, "width": 200, "height": 400}]}}
```

## 11. Working Example with Current Authentication Limitations

Due to staging environment authorization configuration, here's a complete working example:

### Step 1: Create Session
```bash
export EYEPOP_URL="https://compute.staging.eyepop.xyz"
export EYEPOP_SECRET_KEY="your_jwt_token_here"

RESPONSE=$(curl -s -X POST "${EYEPOP_URL}/v1/session" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"user_uuid": ""}')

SESSION_ENDPOINT=$(echo $RESPONSE | jq -r '.session_endpoint')
echo "Session endpoint: $SESSION_ENDPOINT"
```

### Step 2: Create Empty Pipeline (WORKS)
```bash
PIPELINE_RESPONSE=$(curl -s -X POST "${SESSION_ENDPOINT}/pipelines" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "pop": {"components": []},
    "source": {"sourceType": "NONE"},
    "idleTimeoutSeconds": 60,
    "logging": ["out_meta"],
    "videoOutput": "no_output"
  }')

PIPELINE_ID=$(echo $PIPELINE_RESPONSE | jq -r '.id')
echo "Pipeline ID: $PIPELINE_ID"
```

### Step 3: Upload Image (WORKS with empty pipeline)
```bash
curl -X POST "${SESSION_ENDPOINT}/pipelines/${PIPELINE_ID}/source?mode=queue&processing=sync" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Accept: application/jsonl" \
  -F "file=@./examples/example.jpg"
```

### Step 4: What Currently FAILS
```bash
# This fails with "Missing Authorization header" error:
curl -X PATCH "${SESSION_ENDPOINT}/pipelines/${PIPELINE_ID}/pop" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "components": [{
      "type": "inference",
      "inferenceTypes": ["object_detection"],
      "model": "eyepop.person:latest",
      "categoryName": "person",
      "confidenceThreshold": 0.5
    }]
  }'

# This also fails:
curl -X POST "${SESSION_ENDPOINT}/pipelines" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "pop": {
      "components": [{
        "type": "inference",
        "inferenceTypes": ["object_detection"],
        "model": "eyepop.person:latest",
        "categoryName": "person"
      }]
    },
    "source": {"sourceType": "NONE"},
    "idleTimeoutSeconds": 60
  }'
```

### Root Cause Analysis of Authorization Issue

The authorization issue has been identified! The problem occurs when pipeline creation includes **model aliases** (like `"eyepop.person:latest"`) in Pop components.

**The Issue Flow:**
1. POST `/pipelines` with inference components succeeds through JWT authentication
2. `StartPipeline()` calls `internal.ResolvePop()` for Pop components
3. `ResolvePop()` calls `modelService.ResolveAliases()` to resolve model aliases
4. `ResolveAliases()` calls `securityService.GetEyePopAuthorizationHeader()`
5. `GetEyePopAuthorizationHeader()` fails because staging lacks system secret key configuration

**Code Path:**
```
routes/pipelines.go:StartPipeline() 
→ internal/pop_to_pipeline.go:ResolvePop()
→ internal/model_service.go:ResolveAliases() 
→ internal/security_service.go:GetEyePopAuthorizationHeader()
→ Error: "missing user authorization header or eyepop system secret key"
```

**What Works vs. What Fails:**

✅ **Working:**
- Empty pipeline creation (no model alias resolution needed)
- File upload to empty pipelines  
- Debug endpoints
- Any operation not requiring model alias resolution

❌ **Failing:**
- Pipeline creation with model aliases (`"eyepop.person:latest"`)
- Pop component updates with model aliases
- Any inference operation requiring external model resolution

**Solution:** The issue is **account access mismatch**. Your JWT grants access to specific accounts, but the `eyepop.person:latest` model is owned by a different account. 

**Options to resolve:**
1. **Use model UUIDs instead of aliases** - Bypass alias resolution entirely
2. **Add missing account grant** - Include access to the account owning `eyepop.person`
3. **Use different model aliases** - Use models from accounts you have access to
4. **Request universal access** - Add `"target": "all"` grant for broader access

**Current JWT grants access to accounts:**
```
account:034cb8e37f5444e98a78f1be65fd0bff
account:de74d387b5f54fbdbce2f4e4d69e25ce  
account:c78c0119b3cb457a90a30f6d43f2d6f4
account:c8570c5b3f2345e7b25afb74494918b7
account:298218bdb56848efba03a57b1e131cf2
account:49326f2e085a46c39ba73f91c52e436c
account:f37fb2cbce744e83adf4c718ab094a99
```

But `eyepop.person:latest` is likely owned by the main EyePop system account, which requires additional grants.

## 12. Cleanup

Delete pipeline when done:

```bash
curl -X DELETE "${SESSION_ENDPOINT}/pipelines/${PIPELINE_ID}" \
  -H "Authorization: Bearer ${EYEPOP_SECRET_KEY}"
```