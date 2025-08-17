# Suggested Commands for EyePop CLI Development

## Build Commands
- `task build` - Build the CLI binary to ./bin/eyepop
- `go build -o eyepop-cli main.go` - Direct build command

## Run Commands
- `task run` - Run the CLI with arguments
- `./eyepop-cli` - Run the built binary directly
- `go run main.go` - Run without building

## Testing Commands
- `task test:auth` - Test authentication (requires AUTH0_JWT token)
- `go test ./...` - Run all Go tests (when available)

## Development Tools
- `gofmt -w .` - Format Go code
- `go fmt ./...` - Format all Go packages
- `go vet ./...` - Run Go static analysis
- `go mod tidy` - Clean up module dependencies

## Git Commands (Darwin/macOS)
- `git status` - Check repository status
- `git diff` - View uncommitted changes
- `git add .` - Stage changes
- `git commit -m "message"` - Commit changes
- `git push origin feat/tui` - Push to current branch

## System Utilities (macOS)
- `ls -la` - List all files with details
- `pwd` - Print working directory
- `cd` - Change directory
- `grep -r "pattern" .` - Search for pattern recursively
- `find . -name "*.go"` - Find Go files

## Authentication
- `./eyepop-cli auth` - Manage authentication
- `./eyepop-cli auth login` - Login to EyePop account

## Devbox Commands
- `devbox shell` - Enter development environment
- `devbox run test` - Run test script (currently not configured)