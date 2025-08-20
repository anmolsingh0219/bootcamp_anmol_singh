# Homework 5: Data Storage

## Overview
This homework demonstrates data storage practices with multiple formats, environment-driven configuration, and utility functions for flexible data I/O operations.

## Data Storage

### Folder Structure
```
homework5/
├── data/
│   ├── raw/                 # Raw data files (CSV format)
│   └── processed/           # Processed data files (Parquet format)
├── notebooks/
│   └── stage05_data-storage_homework-starter.ipynb
└── README.md
```

### Storage Strategy

#### data/raw/ - Raw Data Storage
- **Format**: CSV files
- **Purpose**: Store unprocessed, original data
- **Characteristics**:
  - Human-readable format
  - Widely compatible
  - Good for data inspection and debugging
  - Preserves original structure

#### data/processed/ - Processed Data Storage  
- **Format**: Parquet files
- **Purpose**: Store cleaned, processed data for analysis
- **Characteristics**:
  - Binary columnar format
  - Efficient compression
  - Fast read/write operations
  - Preserves data types automatically
  - Smaller file sizes

### Environment Configuration
The application uses environment variables for flexible path configuration:

```bash
# .env file
DATA_DIR_RAW=notebooks/data/raw
DATA_DIR_PROCESSED=notebooks/data/processed
```

**Benefits:**
- Easy deployment across different environments
- Configurable without code changes
- Supports different storage backends (local, cloud, etc.)
- Centralized configuration management

### Format Selection Rationale

| Format | Use Case | Advantages | Disadvantages |
|--------|----------|------------|---------------|
| **CSV** | Raw data, sharing, debugging | Human readable, universal compatibility | Larger size, no type preservation |
| **Parquet** | Analytics, processed data | Fast I/O, compression, type safety | Binary format, requires special libraries |

### Utility Functions

The homework implements flexible I/O utilities:

#### `write_df(df, path)`
- Auto-detects format from file extension
- Creates directories if missing
- Handles missing Parquet engines gracefully

#### `read_df(path)`  
- Auto-detects format from file extension
- Handles date parsing for CSV files
- Provides clear error messages for missing dependencies

#### `detect_format(path)`
- Routes operations based on file extension
- Supports: `.csv`, `.parquet`, `.pq`, `.parq`

### Data Validation
Each reload operation validates:
- ✅ Shape consistency (rows × columns)
- ✅ Column names match
- ✅ Data types preserved correctly
- ✅ Date columns maintain datetime format
- ✅ Numeric columns maintain numeric types

### Dependencies
```bash
# Core requirements
pandas>=1.5.0
python-dotenv>=0.19.0

# For Parquet support (optional)
pyarrow>=10.0.0
# OR
fastparquet>=0.8.0
```

### Usage Example
```python
# Load environment configuration
load_dotenv()
RAW = pathlib.Path(os.getenv('DATA_DIR_RAW', 'data/raw'))
PROC = pathlib.Path(os.getenv('DATA_DIR_PROCESSED', 'data/processed'))

# Save in multiple formats
df.to_csv(RAW / "data.csv")
df.to_parquet(PROC / "data.parquet")

# Use utilities for flexible I/O
write_df(df, "data/raw/output.csv")
df_loaded = read_df("data/processed/output.parquet")
```

## Assignment Completion

### ✅ Task 1: Save in Two Formats
- Uses real Alpha Vantage AAPL stock data (101 rows)
- Saves CSV to `data/raw/` with timestamps
- Saves Parquet to `data/processed/` with error handling
- Uses environment variables for directory paths

### ✅ Task 2: Reload and Validate
- Comprehensive validation function checks:
  - Shape equality
  - Column name consistency  
  - Data type preservation
  - Date/numeric type validation
- Clear visual feedback with ✓/✗ indicators

### ✅ Task 3: Utility Functions
- Format detection by file extension
- Automatic directory creation
- Graceful Parquet engine handling
- Consistent error messages and user guidance

### ✅ Task 4: Documentation
- Complete README with storage strategy
- Environment variable usage explained
- Format selection rationale
- Clear setup instructions

## Security Notes
- `.env` files are excluded from version control
- Use `.env.example` as template for required variables
- Sensitive data should never be committed to repositories
