# EyePop CLI Project Overview

## Purpose
The EyePop CLI is a command-line interface tool for interacting with EyePop AI services. It provides authentication, API access, and various utilities for working with the EyePop platform.

## Tech Stack
- **Language**: Go 1.23.6
- **CLI Framework**: Cobra (command-line interface framework)
- **Configuration**: Viper (configuration management)
- **Build System**: Task (Taskfile.yaml for task automation)
- **Development Environment**: Devbox

## Project Structure
```
eyepop-cli/
├── cmd/                  # CLI commands
│   ├── auth.go          # Authentication command
│   ├── login.go         # Login subcommand
│   └── root.go          # Root command and configuration
├── pkg/                 # Package libraries
│   ├── auth/           # Authentication logic
│   ├── clients/        # API clients
│   └── tui/            # Terminal UI components
├── main.go             # Entry point
├── go.mod              # Go module dependencies
├── Taskfile.yaml       # Task automation
└── devbox.json         # Development environment config
```

## Main Components
- Authentication system for EyePop services
- API client implementations
- Terminal UI components for interactive features
- Configuration management using Viper

## Configuration
- Default config file: `$HOME/.eyepop-cli.yaml`
- Supports multiple output formats (text, json)
- Environment variables loaded from `.env` file via devbox