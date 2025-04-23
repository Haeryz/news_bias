import re
import requests
from bs4 import BeautifulSoup
from dateutil import parser
import time
import sys
import random
import os

def scrape_jacobin(event, debug=False, ignore_date=True, fallback_mode=False, limit=None):
    """
    Scrape articles from Jacobin.
    
    Args:
        event (dict): Event info with name, keywords, date_from, date_to
        debug (bool): Enable verbose debug output
        ignore_date (bool): Whether to ignore date restrictions
        fallback_mode (bool): Whether to operate in fallback mode
        limit (int): Maximum number of articles to collect in fallback mode
    
    Returns:
        list: List of article dictionaries
    """
    articles = []
    
    # Get debug flag from sys.argv if not explicitly passed
    if not debug and "--debug" in sys.argv:
        debug = True
    
    # Set maximum articles based on fallback mode
    max_articles = limit if fallback_mode and limit else 200
    
    # Base URL for Jacobin
    base_url = "https://jacobin.com"
    
    # Use a broader, more generic keyword for better search results
    original_keywords = event["keywords"].split(" OR ")[0].lower()
    
    # Map complex topics to simpler search terms that will yield more results
    keyword_mapping = {
        "covid-19": "covid",
        "coronavirus": "covid", 
        "pandemic": "covid",
        "us election": "election",
        "presidential election": "election",
        "biden trump election": "election",
        "george floyd": "protests",
        "black lives matter": "protests",
        "blm": "protests",
        "afghanistan withdrawal": "afghanistan",
        "russia ukraine war": "ukraine",
        "economic recovery": "economy",
        "post-covid economy": "economy",
        "inflation supply chain": "economy",
        "capitol riot": "january 6",
        "capitol insurrection": "january 6",
        "trump indictment": "trump",
        "trump legal": "trump",
        "trump trial": "trump",
        "israel hamas war": "palestine",
        "gaza conflict": "palestine",
        "october 7 attack": "palestine",
        "trump election": "trump",
        "trump victory": "trump"
    }
    
    # Find simpler search keyword
    search_keyword = original_keywords
    for key, value in keyword_mapping.items():
        if key in original_keywords:
            search_keyword = value
            break
    
    # Define search URL with the correct query parameter
    search_url = f"{base_url}/search"
    
    if debug:
        print(f"[JACOBIN DEBUG] Starting scrape for '{original_keywords}' from {event['date_from']} to {event['date_to']}")
        print(f"[JACOBIN DEBUG] Using simplified search term: '{search_keyword}'")
        print(f"[JACOBIN DEBUG] Using search URL: {search_url}?query={search_keyword}")
        print(f"[JACOBIN DEBUG] Date filtering: {'DISABLED' if ignore_date else 'ENABLED'}")
        print(f"[JACOBIN DEBUG] Fallback mode: {'ENABLED' if fallback_mode else 'DISABLED'}")
        print(f"[JACOBIN DEBUG] Target article count: {max_articles}")
    
    # Set up headers to make our requests look like they come from a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://jacobin.com/"
    }
    
    seen_urls = set()
    
    # Try multiple search terms to get more articles
    search_terms = [search_keyword]
    
    # In fallback mode, use very general topics
    if fallback_mode:
        search_terms = ["politics", "economy", "socialism", "capitalism", "labor", "climate", 
                      "democracy", "rights", "justice", "culture", "history", "economics"]
        
        if debug:
            print(f"[JACOBIN DEBUG] Using fallback search terms: {search_terms}")
    # Add additional related terms if search_keyword is in this special list
    elif search_keyword in ["covid", "election", "protests", "economy"]:
        if search_keyword == "covid":
            search_terms.extend(["pandemic", "lockdown", "vaccines"])
        elif search_keyword == "election":
            search_terms.extend(["biden", "trump", "vote"])
        elif search_keyword == "protests":
            search_terms.extend(["floyd", "police", "racial"])
        elif search_keyword == "economy":
            search_terms.extend(["inflation", "recession", "workers"])
    
    for term in search_terms:
        if len(articles) >= max_articles:
            break
            
        if debug:
            print(f"[JACOBIN DEBUG] Searching for term: '{term}'")
        
        try:
            # Construct the search URL with the correct parameter
            params = {"query": term}
            
            # Make the request with a reasonable timeout
            response = requests.get(search_url, params=params, headers=headers, timeout=20)
            
            if response.status_code != 200:
                if debug:
                    print(f"[JACOBIN DEBUG] Search request failed: Status {response.status_code}")
                    print(f"[JACOBIN DEBUG] Headers: {response.headers}")
                    print(f"[JACOBIN DEBUG] Content preview: {response.text[:500]}")
                else:
                    print(f"Jacobin search failed for '{term}': {response.status_code}")
                continue
            
            # Parse the search results with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            
            if debug:
                # Save the search results HTML for inspection
                debug_dir = "debug_output"
                os.makedirs(debug_dir, exist_ok=True)
                with open(f"{debug_dir}/jacobin_search_{term.replace(' ', '_')}.html", "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"[JACOBIN DEBUG] Saved search results to {debug_dir}/jacobin_search_{term.replace(' ', '_')}.html")
                
                # Inspect the page structure
                print("[JACOBIN DEBUG] Page title:", soup.title.text if soup.title else "No title")
                print(f"[JACOBIN DEBUG] Analyzing page structure:")
                
                # Look at all h2 elements which often contain article titles
                h2_tags = soup.find_all("h2")
                print(f"[JACOBIN DEBUG] Found {len(h2_tags)} h2 tags")
                for i, h2 in enumerate(h2_tags[:3]):
                    print(f"[JACOBIN DEBUG] H2 sample {i}: '{h2.text.strip()}'")
                    
                # Look at all article elements
                article_tags = soup.find_all("article")
                print(f"[JACOBIN DEBUG] Found {len(article_tags)} article tags")
            
            # Extract article links - try several different selectors based on actual page structure
            links = []
            
            # First try common patterns for article links
            article_links = soup.select("article h1 a, article h2 a, .post-title a, .title a")
            if article_links:
                links.extend(article_links)
                if debug:
                    print(f"[JACOBIN DEBUG] Found {len(article_links)} article links with primary selectors")
            
            # If we still don't have links, try broader selectors
            if not links:
                if debug:
                    print("[JACOBIN DEBUG] No article links found, trying broader selectors")
                
                # Look for any links within articles
                article_tags = soup.find_all("article")
                for article_tag in article_tags:
                    a_tags = article_tag.find_all("a")
                    for a in a_tags:
                        href = a.get("href", "")
                        if href and "/" in href and not href.startswith("#") and not "javascript:" in href:
                            links.append(a)
            
            # Still no links? Try even broader selectors
            if not links:
                if debug:
                    print("[JACOBIN DEBUG] Still no article links, trying very broad selectors")
                links = [a for a in soup.find_all("a") 
                        if a.get("href") 
                        and not a.get("href").startswith("#") 
                        and not "javascript:" in a.get("href")
                        and len(a.get_text(strip=True)) > 20]  # Links with substantial text
            
            if not links:
                if debug:
                    print(f"[JACOBIN DEBUG] Could not find any usable links for '{term}'")
                continue
            
            if debug:
                print(f"[JACOBIN DEBUG] Found {len(links)} potential links to process")
            
            # Process each article link
            for link in links:
                if len(articles) >= max_articles:
                    break
                    
                url = link.get("href", "")
                if not url:
                    continue
                
                # Ensure the URL is absolute
                if not url.startswith("http"):
                    url = base_url + (url if url.startswith("/") else f"/{url}")
                
                # Skip if we've already seen this URL
                if url in seen_urls:
                    continue
                    
                seen_urls.add(url)
                
                # Skip if this doesn't look like an article URL
                if "/author/" in url or "/about/" in url or "/store/" in url:
                    continue
                
                if debug:
                    print(f"[JACOBIN DEBUG] Processing article: {url}")
                
                try:
                    # Add a delay to avoid hitting rate limits
                    time.sleep(1 + random.random())
                    
                    article_response = requests.get(url, headers=headers, timeout=20)
                    
                    if article_response.status_code != 200:
                        if debug:
                            print(f"[JACOBIN DEBUG] Failed to retrieve article: {url}, status {article_response.status_code}")
                        continue
                    
                    article_soup = BeautifulSoup(article_response.text, "html.parser")
                    
                    # Look for title in multiple locations
                    title_tag = (article_soup.find("h1") or 
                                article_soup.select_one(".article-headline, .post-title, .entry-title"))
                    
                    # If no title found using selectors, try any h1
                    if not title_tag:
                        h1s = article_soup.find_all("h1")
                        if h1s:
                            title_tag = h1s[0]
                    
                    # Look for article content in multiple locations
                    content_div = (article_soup.select_one("article .content, .article-body, .post-content") or
                                  article_soup.select_one(".entry-content, .article__content"))
                                  
                    # If still no content div, try finding the main article container
                    if not content_div:
                        content_div = article_soup.find("article") or article_soup.select_one("main")
                    
                    # Debug content div
                    if debug and content_div:
                        print(f"[JACOBIN DEBUG] Content div class: {content_div.get('class', 'No class')}")
                        
                    # Debug title
                    if debug:
                        if title_tag:
                            print(f"[JACOBIN DEBUG] Found title: '{title_tag.text.strip()}'")
                        else:
                            print("[JACOBIN DEBUG] No title found")
                    
                    # Skip article if we don't have title or content
                    if not title_tag:
                        if debug:
                            print("[JACOBIN DEBUG] No title found, skipping article")
                        continue
                        
                    title = title_tag.get_text(strip=True)
                    
                    if not content_div:
                        if debug:
                            print("[JACOBIN DEBUG] No content div found, skipping article")
                        continue
                    
                    # Extract content: first try paragraphs, then full text
                    paragraphs = content_div.find_all("p")
                    if paragraphs:
                        content = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
                    else:
                        content = content_div.get_text(strip=True)
                    
                    # Skip if content is too short
                    if len(content) < 100:
                        if debug:
                            print(f"[JACOBIN DEBUG] Content too short ({len(content)} chars), skipping")
                        continue
                    
                    # Look for date in various places
                    date_element = (article_soup.find("time") or
                                  article_soup.select_one(".date, .pubdate, .published, .post-date") or
                                  article_soup.select_one("[itemprop='datePublished']"))
                    
                    date_str = ""
                    if date_element:
                        # Try datetime attribute first, then content
                        date_str = date_element.get("datetime", "") or date_element.get_text(strip=True)
                    
                    # No date element? Try meta tags
                    if not date_str:
                        meta_date = article_soup.find("meta", {"property": "article:published_time"})
                        if meta_date:
                            date_str = meta_date.get("content", "")
                    
                    # Still no date? Try extracting from URL or use current date
                    if not date_str:
                        url_date_match = re.search(r'/(\d{4})/(\d{1,2})/', url)
                        if url_date_match:
                            year, month = url_date_match.groups()
                            date_str = f"{year}-{month.zfill(2)}-01"  # Default to 1st of month
                        else:
                            date_str = time.strftime("%Y-%m-%dT%H:%M:%S")
                    
                    # Parse the date string
                    try:
                        date = parser.parse(date_str).isoformat()
                        if debug:
                            print(f"[JACOBIN DEBUG] Parsed date: '{date}' from '{date_str}'")
                    except Exception as e:
                        if debug:
                            print(f"[JACOBIN DEBUG] Date parsing error: {e}, using current date")
                        date = time.strftime("%Y-%m-%dT%H:%M:%S")
                    
                    # Skip articles outside our date range if we're not ignoring dates
                    if not ignore_date and (date < event["date_from"] or date > event["date_to"]):
                        if debug:
                            print(f"[JACOBIN DEBUG] Article date {date} outside range {event['date_from']} to {event['date_to']}")
                        continue
                    
                    # Create and add the article
                    article = {
                        "title": title,
                        "content": content,
                        "outlet": "Jacobin",
                        "date": date,
                        "event": event["name"],
                        "url": url
                    }
                    
                    articles.append(article)
                    
                    if debug:
                        print(f"[JACOBIN DEBUG] SUCCESS! Added article #{len(articles)}: '{title[:50]}...'")
                    else:
                        print(f"Added Jacobin article: {title[:50]}...")
                
                except Exception as e:
                    if debug:
                        print(f"[JACOBIN DEBUG] Error processing article: {e}")
                    continue
        
        except Exception as e:
            if debug:
                print(f"[JACOBIN DEBUG] Error searching for '{term}': {e}")
            else:
                print(f"Error searching Jacobin for '{term}': {e}")
    
    # If we didn't get enough articles from searching specific topics, try the home page
    if len(articles) < max_articles:
        try:
            if debug:
                print(f"[JACOBIN DEBUG] Not enough articles collected ({len(articles)}). Trying more general approaches.")
            
            # Try getting articles from the home page
            if debug:
                print("[JACOBIN DEBUG] Trying to get articles from home page")
                
            home_response = requests.get(base_url, headers=headers, timeout=20)
            
            if home_response.status_code == 200:
                home_soup = BeautifulSoup(home_response.text, "html.parser")
                
                # Find article links on home page
                home_links = home_soup.select("article a, h2 a, h3 a")
                
                if debug:
                    print(f"[JACOBIN DEBUG] Found {len(home_links)} links on home page")
                
                # Process each link
                for link in home_links:
                    if len(articles) >= max_articles:
                        break
                        
                    url = link.get("href", "")
                    if not url or url in seen_urls:
                        continue
                    
                    # Process this link as an article
                    if not url.startswith("http"):
                        url = base_url + (url if url.startswith("/") else f"/{url}")
                        
                    seen_urls.add(url)
                    
                    # Skip if this doesn't look like an article URL
                    if "/author/" in url or "/about/" in url or "/store/" in url:
                        continue
                    
                    if debug:
                        print(f"[JACOBIN DEBUG] Processing home page article: {url}")
                    
                    try:
                        # Process article the same way as search results
                        # ...abbreviated article processing logic...
                        article_response = requests.get(url, headers=headers, timeout=20)
                        if article_response.status_code != 200:
                            continue
                            
                        article_soup = BeautifulSoup(article_response.text, "html.parser")
                        
                        # Get title
                        title_tag = (article_soup.find("h1") or 
                                    article_soup.select_one(".article-headline, .post-title, .entry-title"))
                        
                        if not title_tag:
                            continue
                            
                        title = title_tag.get_text(strip=True)
                        
                        # Get content
                        content_div = (article_soup.select_one("article .content, .article-body, .post-content") or
                                      article_soup.select_one(".entry-content, .article__content") or
                                      article_soup.find("article"))
                                      
                        if not content_div:
                            continue
                            
                        # Get paragraphs
                        paragraphs = content_div.find_all("p")
                        if paragraphs:
                            content = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
                        else:
                            content = content_div.get_text(strip=True)
                            
                        if len(content) < 100:
                            continue
                            
                        # Always use current date in fallback mode
                        if fallback_mode:
                            date = time.strftime("%Y-%m-%dT%H:%M:%S")
                        else:
                            # Try to get date
                            date_element = (article_soup.find("time") or
                                          article_soup.select_one(".date, .pubdate, .published, .post-date"))
                                          
                            date_str = ""
                            if date_element:
                                date_str = date_element.get("datetime", "") or date_element.get_text(strip=True)
                                
                            if not date_str:
                                date = time.strftime("%Y-%m-%dT%H:%M:%S")
                            else:
                                try:
                                    date = parser.parse(date_str).isoformat()
                                except:
                                    date = time.strftime("%Y-%m-%dT%H:%M:%S")
                        
                        # Add the article
                        article = {
                            "title": title,
                            "content": content,
                            "outlet": "Jacobin",
                            "date": date,
                            "event": event["name"],
                            "url": url
                        }
                        
                        articles.append(article)
                        
                        if debug:
                            print(f"[JACOBIN DEBUG] Added home page article #{len(articles)}: '{title[:50]}...'")
                        else:
                            print(f"Added Jacobin article: {title[:50]}...")
                            
                    except Exception as e:
                        if debug:
                            print(f"[JACOBIN DEBUG] Error processing home page article: {e}")
                        continue
                        
                    # Brief delay
                    time.sleep(1)
            
        except Exception as e:
            if debug:
                print(f"[JACOBIN DEBUG] Error processing home page: {e}")
    
    # If we still don't have enough articles, try category pages
    if len(articles) < max_articles:
        try:
            categories = ["politics", "economy", "culture", "world", "us-politics", "ideology"]
            
            if debug:
                print(f"[JACOBIN DEBUG] Still need more articles. Trying category pages.")
            
            for category in categories:
                if len(articles) >= max_articles:
                    break
                    
                category_url = f"{base_url}/category/{category}"
                
                if debug:
                    print(f"[JACOBIN DEBUG] Trying category: {category_url}")
                
                try:
                    category_response = requests.get(category_url, headers=headers, timeout=20)
                    if category_response.status_code != 200:
                        if debug:
                            print(f"[JACOBIN DEBUG] Failed to access category {category}: {category_response.status_code}")
                        continue
                        
                    category_soup = BeautifulSoup(category_response.text, "html.parser")
                    
                    # Find article links
                    category_links = category_soup.select("article a, h2 a, h3 a")
                    
                    if debug:
                        print(f"[JACOBIN DEBUG] Found {len(category_links)} links in {category} category")
                    
                    # Process each link (same as home page processing)
                    for link in category_links:
                        if len(articles) >= max_articles:
                            break
                            
                        url = link.get("href", "")
                        if not url or url in seen_urls:
                            continue
                        
                        # Process this link as an article
                        if not url.startswith("http"):
                            url = base_url + (url if url.startswith("/") else f"/{url}")
                            
                        seen_urls.add(url)
                        
                        # Skip if this doesn't look like an article URL
                        if "/author/" in url or "/about/" in url or "/store/" in url:
                            continue
                        
                        if debug:
                            print(f"[JACOBIN DEBUG] Processing category article: {url}")
                        
                        try:
                            # Process article the same way as search results
                            article_response = requests.get(url, headers=headers, timeout=20)
                            if article_response.status_code != 200:
                                continue
                                
                            article_soup = BeautifulSoup(article_response.text, "html.parser")
                            
                            # Get title
                            title_tag = (article_soup.find("h1") or 
                                        article_soup.select_one(".article-headline, .post-title, .entry-title"))
                            
                            if not title_tag:
                                continue
                                
                            title = title_tag.get_text(strip=True)
                            
                            # Get content
                            content_div = (article_soup.select_one("article .content, .article-body, .post-content") or
                                          article_soup.select_one(".entry-content, .article__content") or
                                          article_soup.find("article"))
                                          
                            if not content_div:
                                continue
                                
                            # Get paragraphs
                            paragraphs = content_div.find_all("p")
                            if paragraphs:
                                content = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
                            else:
                                content = content_div.get_text(strip=True)
                                
                            if len(content) < 100:
                                continue
                                
                            # Always use current date in fallback mode
                            if fallback_mode:
                                date = time.strftime("%Y-%m-%dT%H:%M:%S")
                            else:
                                # Try to get date
                                date_element = (article_soup.find("time") or
                                              article_soup.select_one(".date, .pubdate, .published, .post-date"))
                                              
                                date_str = ""
                                if date_element:
                                    date_str = date_element.get("datetime", "") or date_element.get_text(strip=True)
                                    
                                if not date_str:
                                    date = time.strftime("%Y-%m-%dT%H:%M:%S")
                                else:
                                    try:
                                        date = parser.parse(date_str).isoformat()
                                    except:
                                        date = time.strftime("%Y-%m-%dT%H:%M:%S")
                            
                            # Add the article
                            article = {
                                "title": title,
                                "content": content,
                                "outlet": "Jacobin",
                                "date": date,
                                "event": event["name"],
                                "url": url
                            }
                            
                            articles.append(article)
                            
                            if debug:
                                print(f"[JACOBIN DEBUG] Added category article #{len(articles)}: '{title[:50]}...'")
                            else:
                                print(f"Added Jacobin article: {title[:50]}...")
                                
                        except Exception as e:
                            if debug:
                                print(f"[JACOBIN DEBUG] Error processing category article: {e}")
                            continue
                            
                        # Brief delay
                        time.sleep(1)
                    
                except Exception as e:
                    if debug:
                        print(f"[JACOBIN DEBUG] Error processing {category} category: {e}")
        except Exception as e:
            if debug:
                print(f"[JACOBIN DEBUG] Error processing categories: {e}")
    
    if debug:
        print(f"[JACOBIN DEBUG] Scraping complete. Collected {len(articles)} articles total")
    
    return articles[:max_articles]
