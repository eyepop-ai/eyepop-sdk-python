# Task Completion Requirements

When completing any coding task in the EyePop CLI project, ensure the following steps are performed:

## 1. Code Formatting
- Run `gofmt -w .` or `go fmt ./...` to format all Go files
- Ensure consistent indentation and spacing

## 2. Static Analysis
- Run `go vet ./...` to check for common Go issues
- Address any warnings or errors reported

## 3. Dependency Management
- Run `go mod tidy` if any dependencies were added or removed
- Ensure go.mod and go.sum are up to date

## 4. Build Verification
- Run `task build` or `go build -o eyepop-cli main.go` to ensure the code compiles
- Verify no compilation errors

## 5. Testing
- Run any relevant tests with `go test ./...` (when test files are available)
- For authentication features, use `task test:auth` if applicable
- Ensure all tests pass before considering task complete

## 6. Manual Testing
- Test the CLI commands manually if applicable
- Verify the expected behavior works correctly

## 7. Git Status Check
- Run `git status` to review all changes
- Ensure only intended files are modified
- No temporary or debug files should be included

## Important Notes
- NEVER commit changes unless explicitly requested by the user
- Always verify the build succeeds before marking task as complete
- If unsure about testing commands, ask the user for clarification
- Consider writing test commands to CLAUDE.md for future reference if provided by user