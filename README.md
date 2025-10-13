# Gold and Silver Rate Scraper & Database Seeder

This project contains scripts to scrape gold and silver rates from fenegosida.org and seed the data into a PostgreSQL database.

## Files

- `scrape_gold_rates.py` - Scrapes rate data and saves to CSV
- `seed_database.py` - Seeds CSV data into PostgreSQL database
- `requirements.txt` - Python dependencies

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your database URL:
```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/database_name"
```

## Usage

### 1. Scrape Data

```bash
python3 scrape_gold_rates.py
```

This will:
- Scrape data for years 2075-2082 and all months
- Create a CSV file with timestamp: `gold_silver_rates_YYYYMMDD_HHMMSS.csv`
- Show real-time progress and write data immediately

### 2. Seed Database

```bash
python3 seed_database.py
```

This will:
- Find the most recent CSV file
- Create the `gold_silver_rates` table if it doesn't exist
- Ask if you want to clear existing data
- Insert/update data with duplicate handling
- Show statistics about the inserted data

## Database Schema

```sql
CREATE TABLE gold_silver_rates (
    id SERIAL PRIMARY KEY,
    date VARCHAR(50) NOT NULL,
    year INTEGER NOT NULL,
    month VARCHAR(20) NOT NULL,
    day INTEGER NOT NULL,
    fine_gold INTEGER,
    standard_gold INTEGER,
    silver DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(year, month, day)
);
```

## CSV Format

The CSV contains these columns:
- `date` - Date string (e.g., "2082-Bhadra-01")
- `year` - Nepali year (e.g., 2082)
- `month` - Nepali month name (e.g., "Bhadra")
- `day` - Day of month (1-32)
- `fine_gold` - Fine gold (9999) price per tola
- `standard_gold` - Standard gold (9950) price per tola
- `silver` - Silver price per tola

## Features

- **Real-time writing** - CSV data is written immediately after each month
- **Duplicate handling** - Database script handles duplicate records gracefully
- **Batch processing** - Efficient bulk inserts for large datasets
- **Error handling** - Robust error handling and logging
- **Statistics** - Shows data quality and price ranges
- **Resumable** - Can interrupt and resume without losing data