# Scraping CLI

A command-line interface for web scraping using CrewAI and Browserbase. Supports scraping from Tesco, Asda, and Costco UK platforms.

## Features

- **Multi-vendor support**: Scrape from Tesco, Asda, and Costco UK
- **CrewAI integration**: Advanced agent orchestration for complex scraping tasks
- **Browserbase integration**: Cloud-based browser automation
- **Flexible output formats**: JSON, CSV, and table formats
- **Configurable logging**: Verbose and quiet modes
- **Modular architecture**: Clean, maintainable codebase

## Installation

### Prerequisites

- Python 3.10 or higher
- Git

### Install from source

```bash
# Clone the repository
git clone https://github.com/scraping-cli/scraping-cli.git
cd scraping-cli

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

### Basic Commands

```bash
# Scrape from Tesco
python scraping_cli.py scrape --vendor tesco --urls "https://www.tesco.com/groceries"

# List previous results
python scraping_cli.py list --format table

# Export results
python scraping_cli.py export --output results.json --format json
```

### Command Options

#### Scrape Command
- `--vendor, -V`: Target vendor platform (tesco, asda, costco)
- `--urls, -u`: URLs to scrape (space-separated)
- `--category, -c`: Product category for specialized scraping
- `--output, -o`: Output file path for results
- `--format, -f`: Output format (json, csv)

#### List Command
- `--format, -f`: Output format (json, csv, table)

#### Export Command
- `--output, -o`: Output file path (required)
- `--format, -f`: Output format (json, csv)

#### Global Options
- `--verbose, -v`: Enable verbose logging

### Examples

```bash
# Scrape multiple URLs from Tesco
python scraping_cli.py scrape --vendor tesco \
  --urls "https://www.tesco.com/groceries" "https://www.tesco.com/food" \
  --category "groceries" \
  --output "tesco_results.json" \
  --format json

# List results in table format
python scraping_cli.py list --format table

# Export results to CSV
python scraping_cli.py export --output "results.csv" --format csv

# Enable verbose logging
python scraping_cli.py --verbose scrape --vendor asda --urls "https://www.asda.com"
```

## Project Structure

```
scraping-cli/
├── scraping_cli/           # Main package
│   ├── __init__.py        # Package initialization
│   ├── parser.py          # Command-line argument parsing
│   ├── config.py          # Configuration management
│   └── logging_config.py  # Logging setup
├── scraping_cli.py        # Main entry point
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Code Style

This project uses:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

```bash
# Format code
black .

# Lint code
flake8 .

# Type check
mypy .
```

## Architecture

The CLI is built with a modular architecture:

- **Parser Module**: Handles command-line argument parsing
- **Configuration Module**: Manages CLI options and validation
- **Logging Module**: Provides configurable logging levels
- **Main Script**: Orchestrates all components

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] CrewAI integration for agent orchestration
- [ ] Browserbase integration for web automation
- [ ] Tesco scraping implementation
- [ ] Asda scraping implementation
- [ ] Costco scraping implementation
- [ ] Async agent deployment system
- [ ] Intelligent load balancing
- [ ] Data validation and cleaning
- [ ] Enhanced results display

## Support

For support, please open an issue on GitHub or contact the development team. 