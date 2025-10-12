import requests
import csv
import re
from bs4 import BeautifulSoup
from datetime import datetime
import time

# Configuration
BASE_URL = "https://www.fenegosida.org/rate-history.php"
YEARS = list(range(2075, 2083))  # 2075 to 2082 (earlier years have no data)
MONTHS = [
    "Baisakh", "Jestha", "Ashad", "Shrawan", "Bhadra", "Ashoj",
    "Kartik", "Mansir", "Poush", "Magh", "Falgun", "Chaitra"
]

def extract_price(text):
    """Extract numeric price from text like 'FINE GOLD (9999): <b>195500</b>'"""
    if not text:
        return None
    
    # Convert to string if it's a BeautifulSoup element
    text_str = str(text)
    
    # Find the price in bold tags - this is the actual price
    bold_match = re.search(r'<b>([0-9.]+)</b>', text_str)
    if bold_match:
        price_str = bold_match.group(1)
        # Handle decimal numbers
        if '.' in price_str:
            return float(price_str)
        else:
            return int(price_str)
    
    return None

def scrape_rates_for_month(year, month):
    """Scrape rates for a specific year and month"""
    print(f"Scraping data for {month} {year}...")
    
    # Prepare POST data
    data = {
        'year': str(year),
        'month': month,
        'submit': 'Submit'
    }
    
    try:
        # Make the POST request
        response = requests.post(BASE_URL, data=data, timeout=30)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Debug: Check if we got valid HTML
        if len(response.content) < 100:
            print(f"Response too short for {month} {year}: {len(response.content)} bytes")
            return []
        
        # Find all rate tables and select the first one (PER 1 TOLA)
        tables = soup.find_all('table', class_='table_rate_month')
        if not tables:
            print(f"No rate tables found for {month} {year}")
            return []
        
        # Use the first table which should be "PER 1 TOLA"
        table = tables[0]
        
        # Check if this is indeed the PER 1 TOLA table
        first_row = table.find('tr')
        if first_row and 'PER 1 TOLA' not in first_row.get_text():
            print(f"First table is not PER 1 TOLA for {month} {year}")
            return []
        
        # Get all rows directly (no tbody in this HTML)
        rows = table.find_all('tr')
        
        if not rows:
            print(f"No rows found in table for {month} {year}")
            return []
        
        rates_data = []
        
        for row in rows:
            # Skip header row
            if row.find('td', {'colspan': '4'}):
                continue
            
            th = row.find('th')
            if not th:
                continue
                
            day = th.text.strip()
            if not day.isdigit():
                continue
                
            day = int(day)
            
            # Extract price data from td elements
            tds = row.find_all('td')
            if len(tds) < 3:
                continue
            
            # Pass the HTML element, not just the text
            fine_gold_html = tds[0] if tds[0] else ""
            standard_gold_html = tds[1] if tds[1] else ""
            silver_html = tds[2] if tds[2] else ""
            
            # Extract prices from HTML (not text)
            fine_gold = extract_price(fine_gold_html)
            standard_gold = extract_price(standard_gold_html)
            silver = extract_price(silver_html)
            
            # Create date string
            date_str = f"{year}-{month}-{day:02d}"
            
            rates_data.append({
                'date': date_str,
                'year': year,
                'month': month,
                'day': day,
                'fine_gold': fine_gold,
                'standard_gold': standard_gold,
                'silver': silver
            })
        
        print(f"Found {len(rates_data)} days of data for {month} {year}")
        return rates_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {month} {year}: {e}")
        return []
    except Exception as e:
        print(f"Error parsing data for {month} {year}: {e}")
        return []

def test_single_month():
    """Test function to check a single month before running full scrape"""
    print("Testing with a single month first...")
    test_data = scrape_rates_for_month(2082, "Bhadra")
    if test_data:
        print(f"Test successful! Found {len(test_data)} records")
        print("Sample record:", test_data[0] if test_data else "None")
        return True
    else:
        print("Test failed - no data found")
        return False

def main():
    """Main function to scrape all data and save to CSV"""
    print("Starting gold and silver rate scraping...")
    
    # Test first
    if not test_single_month():
        print("Aborting due to test failure. Check the website structure.")
        return
    
    print(f"\nWill scrape {len(YEARS)} years × {len(MONTHS)} months = {len(YEARS) * len(MONTHS)} combinations")
    
    # Create output file immediately
    output_file = f"gold_silver_rates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = ['date', 'year', 'month', 'day', 'fine_gold', 'standard_gold', 'silver']
    
    # Statistics tracking
    total_records = 0
    valid_fine_gold_count = 0
    valid_standard_gold_count = 0
    valid_silver_count = 0
    fine_gold_prices = []
    silver_prices = []
    
    total_combinations = len(YEARS) * len(MONTHS)
    current_combination = 0
    
    # Open CSV file and write header
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for year in YEARS:
            for month in MONTHS:
                current_combination += 1
                print(f"\nProgress: {current_combination}/{total_combinations}")
                
                rates = scrape_rates_for_month(year, month)
                
                # Write data immediately to CSV
                if rates:
                    for row in rates:
                        writer.writerow(row)
                        total_records += 1
                        
                        # Update statistics
                        if row['fine_gold'] is not None:
                            valid_fine_gold_count += 1
                            fine_gold_prices.append(row['fine_gold'])
                        if row['standard_gold'] is not None:
                            valid_standard_gold_count += 1
                        if row['silver'] is not None:
                            valid_silver_count += 1
                            silver_prices.append(row['silver'])
                    
                    # Flush to ensure data is written immediately
                    csvfile.flush()
                    print(f"✓ Wrote {len(rates)} records to CSV")
                else:
                    print(f"✗ No data found for {month} {year}")
                
                # Be nice to the server - small delay between requests
                time.sleep(1)
    
    print(f"\nScraping completed!")
    print(f"Total records collected: {total_records}")
    print(f"Data saved to: {output_file}")
    
    # Print final statistics
    print(f"\nData quality:")
    print(f"- Fine Gold records: {valid_fine_gold_count}")
    print(f"- Standard Gold records: {valid_standard_gold_count}")
    print(f"- Silver records: {valid_silver_count}")
    
    if fine_gold_prices:
        print(f"- Fine Gold price range: {min(fine_gold_prices)} - {max(fine_gold_prices)}")
    
    if silver_prices:
        print(f"- Silver price range: {min(silver_prices)} - {max(silver_prices)}")

if __name__ == "__main__":
    main()
