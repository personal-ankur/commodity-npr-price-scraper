import requests
import csv
import re
from bs4 import BeautifulSoup
from datetime import datetime
import time

# Configuration
BASE_URL = "https://www.fenegosida.org/rate-history.php"
YEARS = list(range(2073, 2083))  # 2073 to 2082
MONTHS = [
    "Baisakh", "Jestha", "Ashad", "Shrawan", "Bhadra", "Ashoj",
    "Kartik", "Mansir", "Poush", "Magh", "Falgun", "Chaitra"
]

def extract_price(text):
    """Extract numeric price from text like 'FINE GOLD (9999): <b>195500</b>'"""
    if not text:
        return None
    
    # Find numbers in bold tags or just numbers
    bold_match = re.search(r'<b>(\d+)</b>', str(text))
    if bold_match:
        return int(bold_match.group(1))
    
    # Fallback: find any number in the text
    number_match = re.search(r'\d+', str(text))
    if number_match:
        return int(number_match.group())
    
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
        
        # Find the rate table
        table = soup.find('table', class_='table_rate_month')
        if not table:
            print(f"No rate table found for {month} {year}")
            return []
        
        rates_data = []
        rows = table.find('tbody').find_all('tr')
        
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
            
            fine_gold_text = tds[0].text if tds[0] else ""
            standard_gold_text = tds[1].text if tds[1] else ""
            silver_text = tds[2].text if tds[2] else ""
            
            # Extract prices
            fine_gold = extract_price(fine_gold_text)
            standard_gold = extract_price(standard_gold_text)
            silver = extract_price(silver_text)
            
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

def main():
    """Main function to scrape all data and save to CSV"""
    print("Starting gold and silver rate scraping...")
    print(f"Will scrape {len(YEARS)} years × {len(MONTHS)} months = {len(YEARS) * len(MONTHS)} combinations")
    
    all_data = []
    total_combinations = len(YEARS) * len(MONTHS)
    current_combination = 0
    
    for year in YEARS:
        for month in MONTHS:
            current_combination += 1
            print(f"\nProgress: {current_combination}/{total_combinations}")
            
            rates = scrape_rates_for_month(year, month)
            all_data.extend(rates)
            
            # Be nice to the server - small delay between requests
            time.sleep(1)
    
    # Save to CSV
    output_file = f"gold_silver_rates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['date', 'year', 'month', 'day', 'fine_gold', 'standard_gold', 'silver']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in all_data:
            writer.writerow(row)
    
    print(f"\nScraping completed!")
    print(f"Total records collected: {len(all_data)}")
    print(f"Data saved to: {output_file}")
    
    # Print some statistics
    if all_data:
        valid_fine_gold = [r for r in all_data if r['fine_gold'] is not None]
        valid_standard_gold = [r for r in all_data if r['standard_gold'] is not None]
        valid_silver = [r for r in all_data if r['silver'] is not None]
        
        print(f"\nData quality:")
        print(f"- Fine Gold records: {len(valid_fine_gold)}")
        print(f"- Standard Gold records: {len(valid_standard_gold)}")
        print(f"- Silver records: {len(valid_silver)}")
        
        if valid_fine_gold:
            prices = [r['fine_gold'] for r in valid_fine_gold]
            print(f"- Fine Gold price range: {min(prices)} - {max(prices)}")
        
        if valid_silver:
            prices = [r['silver'] for r in valid_silver]
            print(f"- Silver price range: {min(prices)} - {max(prices)}")

if __name__ == "__main__":
    main()
