# Code Style and Conventions

## Go Language Conventions
- **Go Version**: 1.23.6
- **Module Name**: github.com/eyepop-ai/eyepop-cli

## Code Style
- Standard Go formatting using `gofmt`
- Package-level comments with copyright notice
- CamelCase for exported functions and variables
- camelCase for unexported functions and variables

## Project Structure Patterns
- Commands organized in `cmd/` package
- Business logic separated into `pkg/` subdirectories
- Each command as a separate file in `cmd/`
- Cobra command pattern with `Use`, `Short`, and `Long` descriptions

## Import Organization
- Standard library imports first
- External dependencies second
- Internal packages last
- Groups separated by blank lines

## Command Structure (Cobra)
```go
var cmdName = &cobra.Command{
    Use:   "command",
    Short: "Brief description",
    Long:  `Detailed description`,
    Run: func(cmd *cobra.Command, args []string) {
        // Implementation
    },
}
```

## Error Handling
- Explicit error checking and returning
- Use `os.Exit(1)` for fatal errors in main/Execute
- Proper error propagation through return values

## Configuration
- Viper for configuration management
- Support for config file, environment variables, and flags
- Default config location: `$HOME/.eyepop-cli.yaml`

## Testing Conventions
- Test files named `*_test.go`
- Test functions named `Test*`
- Use table-driven tests where appropriate

## Comments
- No unnecessary comments unless explicitly requested
- Package-level documentation comments
- Function documentation for exported functions only when needed