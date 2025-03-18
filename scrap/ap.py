import requests
from bs4 import BeautifulSoup
from dateutil import parser
import time
import re
import json
from datetime import datetime

def scrape_ap(event, fallback_mode=False, limit=None):
    """
    Scrape articles from Associated Press.
    
    Args:
        event (dict): Event info with name, keywords, date_from, date_to
        fallback_mode (bool): Whether to operate in fallback mode (no date filtering)
        limit (int): Maximum number of articles to collect in fallback mode
    
    Returns:
        list: List of article dictionaries
    """
    articles = []
    keywords = event["keywords"].split(" OR ")[0]  # Use first keyword for simplicity
    
    # Set max articles - use limit when in fallback mode
    max_articles = limit if fallback_mode and limit else 200
    
    # AP search URL
    search_url = f"https://apnews.com/search?q={keywords}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml",
        "Referer": "https://apnews.com/"
    }
    seen_urls = set()
    
    # For fallback mode, use different topic keywords
    if fallback_mode:
        # Define diverse topics to get a broad range of articles
        fallback_topics = [
            "politics", "economy", "business", "technology", 
            "science", "health", "sports", "entertainment",
            "world", "education", "environment", "covid",
            "climate", "government", "elections", "military"
        ]
        print(f"Starting AP fallback scraping for diverse topics. Target: {max_articles} articles")
    else:
        print(f"Starting AP scrape for '{keywords}'")
    
    # First try AP's search function
    try:
        # In fallback mode, try multiple topics
        search_terms = fallback_topics if fallback_mode else [keywords]
        
        for search_term in search_terms:
            if len(articles) >= max_articles:
                break
                
            current_search_url = f"https://apnews.com/search?q={search_term}"
            print(f"Searching AP for term: '{search_term}'")
            
            # Try up to 5 pages per term in fallback mode, 10 otherwise
            max_pages = 5 if fallback_mode else 10
            
            for page in range(1, max_pages + 1):
                if len(articles) >= max_articles:
                    break
                    
                current_url = f"{current_search_url}&page={page}"
                print(f"Scraping AP search page {page} for '{search_term}'...")
                
                response = requests.get(current_url, headers=headers, timeout=15)
                if response.status_code != 200:
                    print(f"AP response error: {response.status_code}")
                    break
                    
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Try different possible selectors for AP search results
                search_results = soup.select("a[href*='/article/'], a.Component-link, article a.headline")
                
                if not search_results:
                    print("No article links found on this page. Trying alternative selectors...")
                    links = soup.select("a")
                    search_results = [link for link in links if "/article/" in link.get("href", "")]
                
                if not search_results:
                    print(f"No results found on page {page}")
                    break
                    
                print(f"Found {len(search_results)} potential articles on page {page}")
                
                for result in search_results:
                    if len(articles) >= max_articles:
                        break
                        
                    try:
                        # Get the URL
                        url = result.get("href", "")
                        if not url:
                            continue
                            
                        if not url.startswith("http"):
                            url = f"https://apnews.com{url}"
                            
                        if url in seen_urls:
                            continue
                            
                        seen_urls.add(url)
                        print(f"Processing AP article: {url}")
                        
                        article_response = requests.get(url, headers=headers, timeout=15)
                        if article_response.status_code != 200:
                            print(f"Failed to retrieve article: {url} - Status {article_response.status_code}")
                            continue
                            
                        article_soup = BeautifulSoup(article_response.text, "html.parser")
                        
                        # Get title
                        title_tag = article_soup.select_one("h1") or article_soup.select_one(".headline")
                        if not title_tag:
                            print("Title not found, skipping article")
                            continue
                        
                        title = title_tag.get_text(strip=True)
                        
                        # Try different content selectors
                        content_paragraphs = article_soup.select(".RichTextStoryBody p") or \
                                            article_soup.select(".Article-content p") or \
                                            article_soup.select("article p") or \
                                            article_soup.select(".story-text p")
                        
                        if not content_paragraphs:
                            print("Content paragraphs not found, skipping article")
                            continue
                        
                        content = " ".join([p.get_text(strip=True) for p in content_paragraphs])
                        
                        # If content is too short, likely not a real article
                        if len(content) < 100:
                            print(f"Content too short ({len(content)} chars), skipping")
                            continue
                        
                        # Try to get date (but don't filter by it)
                        article_date = None
                        date_str = ""
                        
                        # First check meta tags
                        meta_published = article_soup.find("meta", {"property": "article:published_time"}) or \
                                        article_soup.find("meta", {"name": "publication_date"})
                        if meta_published:
                            date_str = meta_published.get("content", "")
                            if date_str:
                                try:
                                    article_date = parser.parse(date_str)
                                    print(f"Found meta published time: {date_str}")
                                except Exception as e:
                                    print(f"Error parsing meta date: {e}")
                        
                        # If no date from meta, check JSON-LD
                        if not article_date:
                            script_tags = article_soup.find_all("script", {"type": "application/ld+json"})
                            for script in script_tags:
                                try:
                                    data = json.loads(script.string)
                                    if isinstance(data, dict):
                                        for date_field in ["datePublished", "dateCreated", "dateModified"]:
                                            if date_field in data:
                                                try:
                                                    date_str = data[date_field]
                                                    article_date = parser.parse(date_str)
                                                    print(f"Found {date_field} in JSON-LD: {date_str}")
                                                    break
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass
                        
                        # Fallback to URL extraction
                        if not article_date:
                            url_date_match = re.search(r'/(\d{4})[-/](\d{2})[-/](\d{2})/', url)
                            if url_date_match:
                                year, month, day = url_date_match.groups()
                                try:
                                    article_date = datetime(int(year), int(month), int(day))
                                    print(f"Using date from URL: {article_date.isoformat()}")
                                except Exception as e:
                                    print(f"Failed to use URL date: {e}")
                            else:
                                print("No date found, using current date")
                                article_date = datetime.now()
                        
                        # Create the article object - NEVER filter by date
                        article = {
                            "title": title,
                            "content": content,
                            "outlet": "AP News",
                            "date": article_date.isoformat() if article_date else datetime.now().isoformat(),
                            "event": event["name"],
                            "url": url
                        }
                        
                        articles.append(article)
                        print(f"Added AP article: {title[:40]}... ({len(articles)}/{max_articles})")
                        
                    except Exception as e:
                        print(f"Error processing article: {str(e)}")
                    
                    time.sleep(1)  # Be respectful
                    
                # Add delay between pages
                time.sleep(2)
    
    except Exception as e:
        print(f"Error during AP search: {str(e)}")
    
    # If we didn't get enough articles, try topic-specific hubs
    if len(articles) < max_articles:
        # Standard topic hubs
        topic_hubs = {
            "COVID-19": "coronavirus-pandemic",
            "election": "election-2020",
            "George Floyd": "death-of-george-floyd",
            "Afghanistan": "afghanistan",
            "Ukraine": "russia-ukraine",
            "Capitol": "capitol-siege",
            "Trump": "donald-trump",
            "Israel": "israel-hamas-war",
            "Hamas": "israel-hamas-war",
            "Gaza": "israel-hamas-war",
            "inflation": "inflation",
            "economy": "economy",
        }
        
        # Additional hubs for fallback mode
        if fallback_mode:
            topic_hubs.update({
                "politics": "politics",
                "health": "health",
                "technology": "technology",
                "science": "science",
                "climate": "climate-and-environment",
                "sports": "sports",
                "entertainment": "entertainment",
                "business": "business"
            })
        
        # In fallback mode, try all hubs; otherwise filter to relevant ones
        hub_items = topic_hubs.items() if fallback_mode else [
            (k, v) for k, v in topic_hubs.items() 
            if k.lower() in event["keywords"].lower()
        ]
        
        for keyword, hub in hub_items:
            if len(articles) >= max_articles:
                break
                
            try:
                print(f"Trying AP topic hub: {hub}")
                hub_url = f"https://apnews.com/hub/{hub}"
                response = requests.get(hub_url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    hub_articles = soup.select("a.headline, a.Component-link, a[data-key='card-headline']")
                    
                    print(f"Found {len(hub_articles)} potential articles in {hub} hub")
                    
                    for result in hub_articles:
                        if len(articles) >= max_articles:
                            break
                            
                        url = result.get("href", "")
                        if not url:
                            continue
                            
                        if not url.startswith("http"):
                            url = f"https://apnews.com{url}"
                            
                        if url in seen_urls:
                            continue
                            
                        seen_urls.add(url)
                        print(f"Processing hub article: {url}")
                        
                        # Process article - similar to above
                        try:
                            article_response = requests.get(url, headers=headers, timeout=15)
                            if article_response.status_code != 200:
                                continue
                                
                            article_soup = BeautifulSoup(article_response.text, "html.parser")
                            
                            title_tag = article_soup.select_one("h1") or article_soup.select_one(".headline")
                            if not title_tag:
                                continue
                            
                            title = title_tag.get_text(strip=True)
                            
                            content_paragraphs = article_soup.select(".RichTextStoryBody p") or \
                                                article_soup.select(".Article-content p") or \
                                                article_soup.select("article p") or \
                                                article_soup.select(".story-text p")
                            
                            if not content_paragraphs:
                                continue
                            
                            content = " ".join([p.get_text(strip=True) for p in content_paragraphs])
                            
                            if len(content) < 100:
                                continue
                            
                            # Try to get date (but don't filter by it)
                            article_date = None
                            meta_published = article_soup.find("meta", {"property": "article:published_time"})
                            if meta_published:
                                date_str = meta_published.get("content", "")
                                if date_str:
                                    try:
                                        article_date = parser.parse(date_str)
                                    except:
                                        pass
                            
                            if not article_date:
                                article_date = datetime.now()
                            
                            # Create article with NO date filtering
                            article = {
                                "title": title,
                                "content": content,
                                "outlet": "AP News",
                                "date": article_date.isoformat(),
                                "event": event["name"],
                                "url": url
                            }
                            
                            articles.append(article)
                            print(f"Added hub article: {title[:40]}... ({len(articles)}/{max_articles})")
                            
                        except Exception as e:
                            print(f"Error processing hub article: {str(e)}")
                        
                        time.sleep(1)  # Be respectful
            except Exception as e:
                print(f"Error processing hub {hub}: {str(e)}")
    
    print(f"Total AP articles collected: {len(articles)}")
    return articles
