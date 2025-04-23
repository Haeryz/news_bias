import time
import pandas as pd
from scrap.guardian import scrape_guardian
from scrap.bbc import scrape_bbc
from scrap.ap import scrape_ap
from scrap.jacobin import scrape_jacobin
from scrap.foxnews import scrape_foxnews
from scrap.breitbart import scrape_breitbart
import argparse
import json
import os
from datetime import datetime

# Define events
events = [
    {"name": "COVID-19 Pandemic", "keywords": "COVID-19 OR coronavirus OR pandemic", "date_from": "2020-01-01", "date_to": "2022-12-31"},
    {"name": "2020 U.S. Election", "keywords": "US election OR presidential election OR Biden Trump election", "date_from": "2020-09-01", "date_to": "2020-12-31"},
    {"name": "George Floyd Protests", "keywords": "George Floyd OR BLM OR Black Lives Matter OR protests", "date_from": "2020-05-01", "date_to": "2020-12-31"},
    {"name": "Afghanistan Withdrawal", "keywords": "Afghanistan withdrawal OR US troops Afghanistan OR Taliban takeover", "date_from": "2021-08-01", "date_to": "2021-12-31"},
    {"name": "Russia-Ukraine War", "keywords": "Russia Ukraine war OR Ukraine conflict", "date_from": "2022-02-01", "date_to": "2022-12-31"},
    {"name": "Economic Recovery Post-COVID", "keywords": "economic recovery OR post-COVID economy OR inflation supply chain", "date_from": "2021-01-01", "date_to": "2023-12-31"},
    {"name": "Capitol Riot", "keywords": "Capitol riot OR January 6 OR Capitol insurrection", "date_from": "2021-01-06", "date_to": "2021-03-31"},
    {"name": "Trump Indictments", "keywords": "Trump indictment OR Trump legal OR Trump trial", "date_from": "2023-03-01", "date_to": "2024-03-31"},
    {"name": "Israel-Hamas War", "keywords": "Israel Hamas war OR Gaza conflict OR October 7 attack", "date_from": "2023-10-07", "date_to": "2025-03-08"},
    {"name": "Trump Election", "keywords": "Trump election OR Trump victory", "date_from": "2024-11-01", "date_to": "2025-03-04"}
]

# Define outlets and their scraping functions
outlets = {
    "guardian": {"name": "The Guardian", "function": scrape_guardian, "needs_api_key": True},
    "ap": {"name": "Associated Press", "function": scrape_ap, "needs_api_key": False},
    "bbc": {"name": "BBC News", "function": scrape_bbc, "needs_api_key": False},
    "jacobin": {"name": "Jacobin", "function": scrape_jacobin, "needs_api_key": False},
    "foxnews": {"name": "Fox News", "function": scrape_foxnews, "needs_api_key": False},
    "breitbart": {"name": "Breitbart", "function": scrape_breitbart, "needs_api_key": False}
}

def save_articles(articles, outlet_name, event_name=None, final_save=False):
    """
    Save articles to both JSON and Parquet formats
    
    Args:
        articles: List of article dictionaries to save
        outlet_name: Name of the outlet (used for filenames)
        event_name: Optional event name for JSON file organization
        final_save: Whether this is the final save (for consolidated Parquet)
    """
    if not articles and not final_save:
        print(f"No articles to save for {outlet_name}")
        return
        
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save as JSON (per event)
    if event_name:
        json_filename = f"data/{outlet_name}_{event_name.replace(' ', '_')}.json"
        
        # Load existing data if available
        existing_articles = []
        if os.path.exists(json_filename):
            try:
                with open(json_filename, 'r', encoding='utf-8') as f:
                    existing_articles = json.load(f)
            except Exception as e:
                print(f"Error loading existing JSON: {e}")
                
        # Add new articles, avoiding duplicates
        url_set = {a.get("url", "") for a in existing_articles}
        for article in articles:
            if article.get("url", "") not in url_set:
                existing_articles.append(article)
                url_set.add(article.get("url", ""))
        
        # Save updated dataset
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(existing_articles, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(existing_articles)} articles to {json_filename}")
    
    # Handle Parquet - only save if this is the final save OR we have articles to add
    if final_save or articles:
        outlet_slug = outlet_name.lower().replace(' ', '_')
        parquet_filename = f"data/{outlet_slug}.parquet"
        
        # If we're doing a final save or the parquet file doesn't exist yet, 
        # we need to consolidate all JSON files for this outlet
        if final_save or not os.path.exists(parquet_filename):
            # Load all articles from all event JSONs for this outlet
            all_articles = []
            url_set = set()
            
            # Find all JSON files for this outlet
            json_files = [f for f in os.listdir("data") if f.startswith(outlet_name) and f.endswith(".json")]
            
            for json_file in json_files:
                try:
                    with open(os.path.join("data", json_file), 'r', encoding='utf-8') as f:
                        event_articles = json.load(f)
                        # Avoid duplicates
                        for article in event_articles:
                            url = article.get("url", "")
                            if url and url not in url_set:
                                all_articles.append(article)
                                url_set.add(url)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
            
            # Save consolidated articles to Parquet
            if all_articles:
                try:
                    df = pd.DataFrame(all_articles)
                    df.to_parquet(parquet_filename, compression="snappy")
                    print(f"Saved consolidated {len(all_articles)} articles to {parquet_filename}")
                except Exception as e:
                    print(f"Error saving consolidated Parquet: {e}")
        else:
            # Just add new articles to existing Parquet
            try:
                # Load existing parquet
                try:
                    existing_df = pd.read_parquet(parquet_filename)
                    # Get list of existing URLs to avoid duplicates
                    existing_urls = set(existing_df["url"].tolist() if "url" in existing_df.columns else [])
                except Exception:
                    existing_df = pd.DataFrame()
                    existing_urls = set()
                
                # Filter out articles with URLs that already exist
                new_articles = [a for a in articles if a.get("url", "") not in existing_urls]
                
                if new_articles:
                    # Create new dataframe with just new articles
                    new_df = pd.DataFrame(new_articles)
                    
                    # Combine with existing
                    if not existing_df.empty:
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    else:
                        combined_df = new_df
                    
                    combined_df.to_parquet(parquet_filename, compression="snappy")
                    print(f"Updated {parquet_filename} with {len(new_articles)} new articles, total: {len(combined_df)}")
            except Exception as e:
                print(f"Error updating Parquet: {e}")
                # Fallback: just save the new articles
                try:
                    df = pd.DataFrame(articles)
                    df.to_parquet(parquet_filename, compression="snappy")
                    print(f"Saved {len(articles)} articles to new {parquet_filename} (fallback mode)")
                except Exception as e2:
                    print(f"Complete failure saving to Parquet: {e2}")

def scrape_outlet(outlet_key, event=None, api_key=None, debug=False):
    """Scrape a specific outlet for one or all events"""
    if outlet_key not in outlets:
        print(f"Unknown outlet: {outlet_key}")
        print(f"Available outlets: {', '.join(outlets.keys())}")
        return []
        
    outlet_info = outlets[outlet_key]
    outlet_name = outlet_info["name"]
    scrape_func = outlet_info["function"]
    needs_api_key = outlet_info["needs_api_key"]
    
    all_articles = []
    events_to_process = [event] if event else events
    
    for evt in events_to_process:
        print(f"Scraping {outlet_name} for event: {evt['name']}")
        try:
            if needs_api_key and api_key:
                # For Guardian
                articles = scrape_func(api_key, evt)
            elif outlet_key == "bbc":
                # Special case for BBC to pass debug flag and ignore date
                articles = scrape_func(evt, debug=debug, ignore_date=True)
            elif outlet_key == "jacobin":
                # Special case for Jacobin to pass debug flag
                articles = scrape_func(evt, debug=debug, ignore_date=True)
            else:
                articles = scrape_func(evt)
                
            print(f"Collected {len(articles)} articles for {evt['name']}")
            
            # Save articles for this specific event
            save_articles(articles, outlet_key, evt['name'])
            all_articles.extend(articles)
            
        except Exception as e:
            print(f"Error scraping {outlet_name} for {evt['name']}: {e}")
    
    # Check if we need to run fallback scraping for BBC, AP, or Jacobin
    target_count = 2000
    if outlet_key in ["ap", "bbc", "jacobin"] and len(all_articles) < target_count:
        needed_articles = target_count - len(all_articles)
        print(f"{outlet_name} has only {len(all_articles)} articles. Running fallback scraping for {needed_articles} more...")
        
        # Create a fallback event
        fallback_event = {
            "name": "Others",
            "keywords": "politics OR economy OR society OR news OR world",  # General terms
            "date_from": "2000-01-01",  # Wider date range
            "date_to": "2025-12-31",
            "fallback": True  # Special flag to indicate fallback mode
        }
        
        try:
            if outlet_key == "ap":
                # Call AP scraper in fallback mode
                fallback_articles = scrape_ap(fallback_event, fallback_mode=True, limit=needed_articles)
            elif outlet_key == "bbc":
                # Create a list of general topics to try for BBC
                general_topics = [
                    "politics", "economy", "health", "science", "technology", 
                    "business", "entertainment", "sports", "world", "education"
                ]
                
                fallback_articles = []
                
                # Try each topic until we have enough articles
                for topic in general_topics:
                    if len(fallback_articles) >= needed_articles:
                        break
                        
                    print(f"Trying BBC fallback topic: {topic}")
                    topic_event = fallback_event.copy()
                    topic_event["keywords"] = topic
                    
                    topic_limit = needed_articles - len(fallback_articles)
                    topic_articles = scrape_bbc(
                        topic_event, 
                        debug=debug, 
                        ignore_date=True, 
                        fallback_mode=True,
                        limit=topic_limit
                    )
                    
                    fallback_articles.extend(topic_articles)
                    
                    # Brief pause between topics
                    time.sleep(2)
            elif outlet_key == "jacobin":
                # Call Jacobin scraper in fallback mode
                fallback_articles = scrape_jacobin(
                    fallback_event, 
                    debug=debug,
                    ignore_date=True, 
                    fallback_mode=True,
                    limit=needed_articles
                )
            
            print(f"Fallback scraping collected {len(fallback_articles)} additional {outlet_name} articles")
            
            # Save fallback articles
            save_articles(fallback_articles, outlet_key, "Others")
            all_articles.extend(fallback_articles)
            
        except Exception as e:
            print(f"Error during fallback scraping for {outlet_name}: {e}")
    
    # After processing all events, do a final save to consolidate everything into Parquet
    save_articles([], outlet_key, None, final_save=True)
    
    return all_articles

def main():
    parser = argparse.ArgumentParser(description='Scrape news articles about events')
    parser.add_argument('--source', help='Scrape only a specific outlet (guardian, ap, bbc, jacobin, foxnews, breitbart)')
    parser.add_argument('--event', type=int, help='Scrape only a specific event (index number, starting at 0)')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug output')
    args = parser.parse_args()
    
    # Guardian API key - update with your actual key
    api_key = "0deeddbb-a747-4a10-b437-de5591f6629a"
    
    if args.debug:
        print("Debug mode enabled")
        print(f"Available outlets: {list(outlets.keys())}")
        print(f"Available events: {[e['name'] for e in events]}")
    
    if args.source:
        # Single outlet mode
        outlet_key = args.source.lower()
        print(f"Testing scraper for {outlet_key}")
        
        # Get specific event if requested
        event = None
        if args.event is not None and 0 <= args.event < len(events):
            event = events[args.event]
            print(f"Scraping single event: {event['name']}")
        
        scrape_outlet(outlet_key, event, api_key, args.debug)
    else:
        # All outlets mode
        for outlet_key in outlets:
            outlet_name = outlets[outlet_key]["name"]
            print(f"Starting scraping for {outlet_name}")
            all_articles = scrape_outlet(outlet_key, None, api_key, args.debug)
            print(f"Completed scraping for {outlet_name}, collected {len(all_articles)} total articles")

if __name__ == "__main__":
    main()