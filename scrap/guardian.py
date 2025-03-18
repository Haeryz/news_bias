import requests
from bs4 import BeautifulSoup
import time

def scrape_guardian(api_key, event):
    """
    Scrape articles from The Guardian using the API.
    
    Args:
        api_key (str): Guardian API key
        event (dict): Event info with name, keywords, date_from, date_to
    
    Returns:
        list: List of article dictionaries
    """
    base_url = "https://content.guardianapis.com/search"
    params = {
        "q": event["keywords"],
        "from-date": event["date_from"],
        "to-date": event["date_to"],
        "page-size": 50,  # Max articles per page
        "show-fields": "body",  # Include article content
        "api-key": api_key
    }
    articles = []
    page = 1
    
    while len(articles) < 200:  # Aim for 200 articles per event
        params["page"] = page
        try:
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code != 200:
                print(f"Guardian API error: {response.status_code}")
                break
            data = response.json()
            results = data["response"]["results"]
            if not results:
                break
            
            for item in results:
                content_html = item["fields"]["body"]
                content = BeautifulSoup(content_html, "html.parser").get_text(strip=True)
                article = {
                    "title": item["webTitle"],
                    "content": content,
                    "outlet": "The Guardian",
                    "date": item["webPublicationDate"],
                    "event": event["name"],
                    "url": item["webUrl"]
                }
                articles.append(article)
                if len(articles) >= 200:
                    break
            page += 1
            if page > data["response"]["pages"]:
                break
            time.sleep(1)  # Be polite to the API
        except Exception as e:
            print(f"Error scraping Guardian: {e}")
            break
    
    return articles