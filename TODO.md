# Technical Debt

## Pydantic Discriminated Union Pattern

**File:** `eyepop/worker/worker_types.py`

**Issue:** The `BaseComponent` class and its subclasses (`ForwardComponent`, `InferenceComponent`, `TrackingComponent`, etc.) use inheritance with a `type` field override for discriminated unions. This is an anti-pattern according to Pydantic documentation.

**Current workaround:** Global pyright suppressions in `pyrightconfig.json`:
```json
"reportIncompatibleVariableOverride": false,
"reportAttributeAccessIssue": false
```

**Recommended fix:** Refactor to define each component model independently without inheritance, each with its own `type: Literal[...]` field. This follows Pydantic best practices for discriminated unions.

**Priority:** Low - current approach works, but the global suppressions could hide legitimate type errors elsewhere.
