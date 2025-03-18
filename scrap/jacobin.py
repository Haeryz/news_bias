import requests
from bs4 import BeautifulSoup
from dateutil import parser
import time

def scrape_jacobin(event):
    """
    Scrape articles from Jacobin.
    
    Args:
        event (dict): Event info with name, keywords, date_from, date_to
    
    Returns:
        list: List of article dictionaries
    """
    articles = []
    base_url = "https://jacobin.com/search"
    keywords = event["keywords"].split(" OR ")[0]  # Use first keyword for simplicity
    params = {"q": keywords, "page": 1}
    headers = {"User-Agent": "Mozilla/5.0"}
    seen_urls = set()
    
    while len(articles) < 200:
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Jacobin response error: {response.status_code}")
                break
            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.select("div.post-item a.post-title")  # Jacobin article links
            if not links:
                break
            
            for link in links:
                url = link["href"]
                if not url.startswith("http"):
                    url = "https://jacobin.com" + url
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                
                article_response = requests.get(url, headers=headers, timeout=10)
                article_soup = BeautifulSoup(article_response.text, "html.parser")
                
                title_tag = article_soup.find("h1", {"class": "post-title"})
                content_div = article_soup.find("div", {"class": "post-content"})
                date_tag = article_soup.find("time")
                
                if not (title_tag and content_div and date_tag):
                    continue
                
                title = title_tag.get_text(strip=True)
                content = content_div.get_text(strip=True)
                date_str = date_tag.get_text(strip=True)
                try:
                    date = parser.parse(date_str).isoformat()
                except:
                    continue
                
                if event["date_from"] <= date <= event["date_to"]:
                    article = {
                        "title": title,
                        "content": content,
                        "outlet": "Jacobin",
                        "date": date,
                        "event": event["name"],
                        "url": url
                    }
                    articles.append(article)
                time.sleep(1)
                if len(articles) >= 200:
                    break
            params["page"] += 1
        except Exception as e:
            print(f"Error scraping Jacobin: {e}")
            break
    
    return articles
