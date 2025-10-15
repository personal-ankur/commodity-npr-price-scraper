#!/usr/bin/env python3
"""
Script to seed CSV data into PostgreSQL database
Reads gold and silver rate data from CSV and inserts into database
"""

import os
import csv
import psycopg2
import psycopg2.extras
from datetime import datetime, date
import sys
import glob
import dotenv
from nepali_calendar_utils import NepaliDateConverter

dotenv.load_dotenv()

# Nepali month names mapping (includes common spelling variations)
NEPALI_MONTHS = {
    "Baisakh": 1, "Jestha": 2, "Ashadh": 3, "Ashad": 3,  # Ashad is alternate spelling
    "Shrawan": 4, "Bhadra": 5, "Ashwin": 6, "Ashoj": 6,  # Ashoj is alternate spelling  
    "Kartik": 7, "Mangsir": 8, "Mansir": 8,  # Mansir is alternate spelling
    "Poush": 9, "Magh": 10, "Falgun": 11, "Chaitra": 12
}

def convert_nepali_to_english_date(nepali_year, nepali_month, nepali_day):
    """
    Convert Nepali date to English date using nepali_calendar_utils library
    """
    try:
        # Get month number from mapping
        month_num = NEPALI_MONTHS.get(nepali_month)
        if not month_num:
            return None
        
        # Use the proper library for accurate conversion
        converter = NepaliDateConverter()
        result = converter.convert_nepali_to_english(nepali_year, month_num, nepali_day)
        
        # Extract date from CustomCalendar object
        if result and hasattr(result, 'year') and hasattr(result, 'month') and hasattr(result, 'day_of_month'):
            return date(result.year, result.month, result.day_of_month)
        
        return None
    
    except (ValueError, TypeError, Exception):
        return None

def get_database_connection():
    """Get database connection from environment variable"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    
    try:
        conn = psycopg2.connect(database_url)
        return conn
    except psycopg2.Error as e:
        raise Exception(f"Failed to connect to database: {e}")

def create_table(conn):
    """Create the gold_silver_rates table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS gold_silver_rates (
        id SERIAL PRIMARY KEY,
        date VARCHAR(50) NOT NULL,
        english_date DATE,
        year INTEGER NOT NULL,
        month VARCHAR(20) NOT NULL,
        day INTEGER NOT NULL,
        fine_gold INTEGER,
        standard_gold INTEGER,
        silver DECIMAL(10,2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(year, month, day)
    );
    
    CREATE INDEX IF NOT EXISTS idx_gold_silver_rates_date ON gold_silver_rates(year, month, day);
    CREATE INDEX IF NOT EXISTS idx_gold_silver_rates_year_month ON gold_silver_rates(year, month);
    """
    
    with conn.cursor() as cursor:
        cursor.execute(create_table_sql)
        
        # Add english_date column if it doesn't exist (migration)
        cursor.execute("""
            ALTER TABLE gold_silver_rates 
            ADD COLUMN IF NOT EXISTS english_date DATE;
        """)
        
        conn.commit()
        print("✓ Table 'gold_silver_rates' created/verified")

def clear_existing_data(conn):
    """Clear existing data from the table"""
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM gold_silver_rates")
        conn.commit()
        print("✓ Cleared existing data from table")

def process_csv_file(csv_file_path, conn):
    """Process a single CSV file and insert data into database"""
    print(f"Processing CSV file: {csv_file_path}")
    
    inserted_count = 0
    skipped_count = 0
    error_count = 0
    
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Collect rows for batch insert with deduplication
        rows_to_insert = []
        seen_keys = set()  # Track unique (year, month, day) combinations within batch
        
        for row_num, row in enumerate(reader, 1):
            try:
                # Validate required fields
                if not all([row.get('date'), row.get('year'), row.get('month'), row.get('day')]):
                    print(f"Warning: Skipping row {row_num} - missing required fields")
                    skipped_count += 1
                    continue
                
                # Convert data types
                year = int(row['year'])
                day = int(row['day'])
                month = row['month'].strip()
                date_str = row['date'].strip()
                
                # Check for duplicates within the current batch
                key = (year, month, day)
                if key in seen_keys:
                    print(f"Warning: Skipping duplicate row {row_num} - {date_str}")
                    skipped_count += 1
                    continue
                
                # Handle None/empty values for prices
                fine_gold = int(row['fine_gold']) if row['fine_gold'] and row['fine_gold'].strip() else None
                standard_gold = int(row['standard_gold']) if row['standard_gold'] and row['standard_gold'].strip() else None
                silver = float(row['silver']) if row['silver'] and row['silver'].strip() else None
                
                # Convert Nepali date to English date
                english_date = convert_nepali_to_english_date(year, month, day)
                
                rows_to_insert.append((
                    date_str, english_date, year, month, day, fine_gold, standard_gold, silver
                ))
                seen_keys.add(key)
                
                # Batch insert every 1000 rows
                if len(rows_to_insert) >= 1000:
                    inserted_count += insert_batch(conn, rows_to_insert)
                    rows_to_insert = []
                    seen_keys = set()  # Reset for next batch
                
            except (ValueError, KeyError) as e:
                print(f"Error processing row {row_num}: {e}")
                error_count += 1
                continue
        
        # Insert remaining rows
        if rows_to_insert:
            inserted_count += insert_batch(conn, rows_to_insert)
    
    return inserted_count, skipped_count, error_count

def insert_batch(conn, rows):
    """Insert a batch of rows using ON CONFLICT to handle duplicates"""
    if not rows:
        return 0
    
    insert_sql = """
    INSERT INTO gold_silver_rates (date, english_date, year, month, day, fine_gold, standard_gold, silver)
    VALUES %s
    ON CONFLICT (year, month, day) 
    DO UPDATE SET 
        date = EXCLUDED.date,
        english_date = EXCLUDED.english_date,
        fine_gold = EXCLUDED.fine_gold,
        standard_gold = EXCLUDED.standard_gold,
        silver = EXCLUDED.silver,
        created_at = CURRENT_TIMESTAMP
    """
    
    try:
        with conn.cursor() as cursor:
            psycopg2.extras.execute_values(
                cursor, insert_sql, rows, template=None, page_size=1000
            )
            conn.commit()
            return len(rows)
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error inserting batch: {e}")
        return 0

def find_csv_files():
    """Find CSV files matching the pattern"""
    pattern = "gold_silver_rates_*.csv"
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No CSV files found matching pattern: {pattern}")
        return []
    
    # Sort by modification time (newest first)
    csv_files.sort(key=os.path.getmtime, reverse=True)
    return csv_files

def get_table_stats(conn):
    """Get statistics about the table data"""
    with conn.cursor() as cursor:
        # Total records
        cursor.execute("SELECT COUNT(*) FROM gold_silver_rates")
        total_records = cursor.fetchone()[0]
        
        # Date range
        cursor.execute("SELECT MIN(year), MAX(year) FROM gold_silver_rates")
        year_range = cursor.fetchone()
        
        # Records with prices
        cursor.execute("SELECT COUNT(*) FROM gold_silver_rates WHERE fine_gold IS NOT NULL")
        fine_gold_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM gold_silver_rates WHERE silver IS NOT NULL")
        silver_records = cursor.fetchone()[0]
        
        # Price ranges
        cursor.execute("SELECT MIN(fine_gold), MAX(fine_gold) FROM gold_silver_rates WHERE fine_gold IS NOT NULL")
        fine_gold_range = cursor.fetchone()
        
        cursor.execute("SELECT MIN(silver), MAX(silver) FROM gold_silver_rates WHERE silver IS NOT NULL")
        silver_range = cursor.fetchone()
        
        return {
            'total_records': total_records,
            'year_range': year_range,
            'fine_gold_records': fine_gold_records,
            'silver_records': silver_records,
            'fine_gold_range': fine_gold_range,
            'silver_range': silver_range
        }

def main():
    """Main function to seed database from CSV files"""
    print("Starting database seeding process...")
    
    # Check for CSV files
    #csv_files = find_csv_files()
    #if not csv_files:
    #    print("No CSV files found. Please run the scraping script first.")
    #    sys.exit(1)
    
    #print(f"Found {len(csv_files)} CSV file(s)")
    csv_files = ["data/prices.csv"]
    for i, file in enumerate(csv_files, 1):
        file_size = os.path.getsize(file) / 1024  # KB
        mod_time = datetime.fromtimestamp(os.path.getmtime(file))
        print(f"  {i}. {file} ({file_size:.1f} KB, modified: {mod_time})")
    
    # Use the most recent CSV file
    csv_file = csv_files[0]
    print(f"\nUsing most recent file: {csv_file}")
    
    try:
        # Connect to database
        print("Connecting to PostgreSQL database...")
        conn = get_database_connection()
        print("✓ Database connection established")
        
        # Create table
        create_table(conn)
        
        # Ask user if they want to clear existing data
        response = input("\nDo you want to clear existing data before seeding? (y/N): ").lower()
        if response in ['y', 'yes']:
            clear_existing_data(conn)
        
        # Process CSV file
        print(f"\nProcessing CSV data...")
        start_time = datetime.now()
        
        inserted, skipped, errors = process_csv_file(csv_file, conn)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\nSeeding completed in {duration:.2f} seconds!")
        print(f"✓ Records inserted/updated: {inserted}")
        print(f"✓ Records skipped: {skipped}")
        print(f"✓ Errors encountered: {errors}")
        
        # Show table statistics
        print("\nDatabase statistics:")
        stats = get_table_stats(conn)
        print(f"- Total records: {stats['total_records']}")
        print(f"- Year range: {stats['year_range'][0]} to {stats['year_range'][1]}")
        print(f"- Fine gold records: {stats['fine_gold_records']}")
        print(f"- Silver records: {stats['silver_records']}")
        
        if stats['fine_gold_range'][0] is not None:
            print(f"- Fine gold price range: {stats['fine_gold_range'][0]} to {stats['fine_gold_range'][1]}")
        
        if stats['silver_range'][0] is not None:
            print(f"- Silver price range: {stats['silver_range'][0]} to {stats['silver_range'][1]}")
        
        conn.close()
        print("\n✓ Database seeding completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
