import requests
from bs4 import BeautifulSoup
from dateutil import parser
import time
import sys
import random

def scrape_bbc(event, debug=False, ignore_date=True, fallback_mode=False, limit=None):
    """
    Scrape articles from BBC News.
    
    Args:
        event (dict): Event info with name, keywords, date_from, date_to
        debug (bool): Enable verbose debug output
        ignore_date (bool): Whether to ignore date restrictions (default: True)
        fallback_mode (bool): Whether to operate in fallback mode for more general topics
        limit (int): Maximum number of articles to collect in fallback mode
    
    Returns:
        list: List of article dictionaries
    """
    articles = []
    base_url = "https://www.bbc.co.uk/search"
    keywords = event["keywords"].split(" OR ")[0]  # Use first keyword for simplicity
    params = {"q": keywords, "page": 1}
    
    # Use a more robust header setup
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive"
    }
    
    # Get debug flag from sys.argv if not explicitly passed
    if not debug and "--debug" in sys.argv:
        debug = True
    
    seen_urls = set()
    attempts = 0
    max_attempts = 3
    max_timeout = 20  # Maximum timeout in seconds
    
    # Progress tracking variables
    total_links_found = 0
    processed_links = 0
    articles_outside_date = 0
    articles_with_missing_data = 0
    
    # Fallback detection variables
    consecutive_no_new_urls = 0
    max_consecutive_no_new_urls = 5
    max_articles = limit if fallback_mode and limit else 200
    
    if debug:
        print(f"[BBC DEBUG] Starting scrape for '{keywords}' from {event['date_from']} to {event['date_to']}")
        print(f"[BBC DEBUG] Search URL: {base_url}?q={keywords}")
        print(f"[BBC DEBUG] Date range filtering: {'DISABLED' if ignore_date else 'ENABLED'}")
        print(f"[BBC DEBUG] Fallback mode: {'ENABLED' if fallback_mode else 'DISABLED'}")
        print(f"[BBC DEBUG] Target article count: {max_articles}")
    
    while len(articles) < max_articles and attempts < max_attempts:
        try:
            if debug:
                print(f"[BBC DEBUG] Requesting page {params['page']}, attempt {attempts+1}/{max_attempts}")
                print(f"[BBC DEBUG] Progress: {len(articles)} articles collected so far")
                
            # Gradually increase timeout if we've had failures
            timeout = 10 + (attempts * 5)
            timeout = min(timeout, max_timeout)
            
            if debug:
                print(f"[BBC DEBUG] Using timeout of {timeout} seconds")
            
            response = requests.get(
                base_url, 
                params=params, 
                headers=headers, 
                timeout=timeout
            )
            
            if response.status_code != 200:
                if debug:
                    print(f"[BBC DEBUG] Response error: {response.status_code}")
                else:
                    print(f"BBC response error: {response.status_code}")
                
                attempts += 1
                time.sleep(2 + attempts * 2)  # Exponential backoff
                continue
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # First try to find direct news links by various patterns
            links = soup.select("a[href*='/news/'], a.ssrcss-rl2iw9-PromoLink, div.ssrcss-1f3bvyz-StyledLink a")
            
            # Try alternative patterns if needed
            if not links:
                if debug:
                    print("[BBC DEBUG] No links with primary selectors, trying alternatives")
                # Look for search result links with a heading
                links = soup.select("div.ssrcss-1pc5x23-PromoWrapper a, div.ssrcss-1efsua-PromoContent a")
            
            # As a last resort, try very broad selectors
            if not links:
                if debug:
                    print("[BBC DEBUG] Still no links, trying broader selectors")
                # Look for anything that might be a news link
                links = soup.select("a[href*='bbc.co.uk/news'], a[href*='bbc.com/news']")
            
            if not links and debug:
                print("[BBC DEBUG] No links found with any selectors")
                
                # If we're getting no results for too long, trigger fallback
                consecutive_no_new_urls += 1
                if consecutive_no_new_urls >= max_consecutive_no_new_urls:
                    if debug:
                        print("[BBC DEBUG] Too many pages with no results. Breaking search loop.")
                    break
                
                params["page"] += 1
                continue
            
            # Check if we're seeing all duplicates
            new_urls_found = 0
            for link in links:
                url = link.get("href", "")
                if url and url not in seen_urls:
                    new_urls_found += 1
            
            # If we're not finding any new URLs, increment our counter
            if new_urls_found == 0:
                consecutive_no_new_urls += 1
                if debug:
                    print(f"[BBC DEBUG] No new URLs found on this page. Consecutive pages without new URLs: {consecutive_no_new_urls}/{max_consecutive_no_new_urls}")
                
                # If we've gone too many pages without new URLs, break
                if consecutive_no_new_urls >= max_consecutive_no_new_urls:
                    if debug:
                        print("[BBC DEBUG] Too many consecutive pages without new URLs. Breaking search loop.")
                    break
            else:
                # Reset the counter if we found new URLs
                consecutive_no_new_urls = 0
            
            total_links_found += len(links)
            
            if debug:
                print(f"[BBC DEBUG] Found {len(links)} potential article links (total: {total_links_found}), {new_urls_found} are new")
            
            # Reset attempts counter since we found links
            attempts = 0
            
            for link in links:
                if len(articles) >= max_articles:
                    break
                
                processed_links += 1
                
                url = link.get("href", "")
                if not url:
                    continue
                    
                if not url.startswith("http"):
                    if url.startswith("/"):
                        url = "https://www.bbc.co.uk" + url
                    else:
                        url = "https://www.bbc.co.uk/" + url
                
                if url in seen_urls:
                    if debug:
                        print(f"[BBC DEBUG] Skipping already seen URL: {url}")
                    continue
                
                seen_urls.add(url)
                
                if debug:
                    print(f"[BBC DEBUG] Processing article #{processed_links}: {url}")
                
                try:
                    # Add jitter to avoid rate limiting
                    time.sleep(0.5 + random.random())
                    
                    article_response = requests.get(url, headers=headers, timeout=timeout)
                    if article_response.status_code != 200:
                        if debug:
                            print(f"[BBC DEBUG] Failed to retrieve article: {url}, status {article_response.status_code}")
                        continue
                    
                    article_soup = BeautifulSoup(article_response.text, "html.parser")
                    
                    # Try different selectors for title
                    title_tag = (article_soup.select_one("[data-component='headline'], [id='main-heading']") or 
                                article_soup.find("h1") or 
                                article_soup.select_one(".story-body__h1") or
                                article_soup.select_one(".vxp-media__headline"))
                    
                    if not title_tag and debug:
                        print("[BBC DEBUG] Title not found with primary selectors, trying secondary selectors")
                        h1_tags = article_soup.find_all("h1")
                        if h1_tags:
                            title_tag = h1_tags[0]
                            print(f"[BBC DEBUG] Found title using fallback: {title_tag.text.strip()}")
                    
                    if not title_tag:
                        if debug:
                            print("[BBC DEBUG] Title not found, skipping article")
                        articles_with_missing_data += 1
                        continue
                    
                    title = title_tag.get_text(strip=True)
                    
                    # IMPROVED CONTENT EXTRACTION:
                    # First try the new BBC format with text-block components
                    content = ""
                    text_blocks = article_soup.select("div[data-component='text-block']")
                    
                    if text_blocks:
                        if debug:
                            print(f"[BBC DEBUG] Found {len(text_blocks)} text blocks")
                        
                        # Extract all paragraphs from all text blocks
                        all_paragraphs = []
                        for block in text_blocks:
                            paragraphs = block.find_all("p")
                            all_paragraphs.extend(paragraphs)
                        
                        if all_paragraphs:
                            content = " ".join(p.get_text(strip=True) for p in all_paragraphs)
                            if debug:
                                print(f"[BBC DEBUG] Extracted content from {len(all_paragraphs)} paragraphs")
                    
                    # If no content yet, try older BBC formats
                    if not content:
                        # Try different selectors for content
                        content_selectors = [
                            "div.ssrcss-11r1m41-RichTextComponentWrapper",  # Modern BBC
                            "div.article__body-content",                    # Older BBC articles
                            "div.story-body__inner",                        # Even older BBC format
                            "div.story-body"                                # Very old BBC format
                        ]
                        
                        for selector in content_selectors:
                            content_div = article_soup.select_one(selector)
                            if content_div:
                                # Get all paragraphs from content div
                                paragraphs = content_div.find_all("p")
                                if paragraphs:
                                    content = " ".join(p.get_text(strip=True) for p in paragraphs)
                                    break
                    
                    # Final fallback - try any paragraph in the main article area
                    if not content:
                        article_body = (article_soup.select_one("main") or 
                                        article_soup.select_one("[role='main']") or
                                        article_soup)
                        if article_body:
                            paragraphs = article_body.select("p")
                            if paragraphs:
                                # Filter out navigation, footer paragraphs by length
                                main_paragraphs = [p for p in paragraphs if len(p.get_text(strip=True)) > 40]
                                if main_paragraphs:
                                    content = " ".join(p.get_text(strip=True) for p in main_paragraphs)
                                    if debug:
                                        print(f"[BBC DEBUG] Used fallback content extraction, found {len(main_paragraphs)} paragraphs")
                    
                    if not content:
                        if debug:
                            print("[BBC DEBUG] Content not found, skipping article")
                        articles_with_missing_data += 1
                        continue
                    
                    if len(content) < 100:
                        if debug:
                            print(f"[BBC DEBUG] Content too short ({len(content)} chars), skipping article")
                        articles_with_missing_data += 1
                        continue
                    
                    # Try different selectors for date
                    date_tag = (article_soup.select_one("time") or 
                               article_soup.select_one("[data-testid='timestamp']") or
                               article_soup.select_one(".date"))
                    
                    if not date_tag:
                        # Try to find any element with datetime attribute
                        date_elements = article_soup.select("[datetime]")
                        if date_elements:
                            date_tag = date_elements[0]
                    
                    # If still no date element found but we're ignoring dates anyway, use current date
                    if not date_tag and ignore_date:
                        if debug:
                            print("[BBC DEBUG] No date found, using current date (since date filtering is disabled)")
                        # Create article with current date
                        article = {
                            "title": title,
                            "content": content,
                            "outlet": "BBC News",
                            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "event": event["name"],
                            "url": url
                        }
                        articles.append(article)
                        if debug:
                            print(f"[BBC DEBUG] SUCCESS! Added article #{len(articles)}: '{title[:40]}...'")
                        else:
                            print(f"Added BBC article #{len(articles)}: {title[:40]}...")
                        continue
                    
                    if not date_tag:
                        if debug:
                            print("[BBC DEBUG] Date not found and date filtering required, skipping article")
                        articles_with_missing_data += 1
                        continue
                    
                    # Try to get datetime from attribute first
                    date_str = date_tag.get("datetime", "") or date_tag.get("data-datetime", "")
                    # If no datetime attribute, try getting the text
                    if not date_str:
                        date_str = date_tag.get_text(strip=True)
                    
                    if debug:
                        print(f"[BBC DEBUG] Found date string: '{date_str}'")
                    
                    try:
                        date = parser.parse(date_str).isoformat()
                        
                        if debug:
                            print(f"[BBC DEBUG] Parsed date to '{date}'")
                        
                        # Check date range only if we're not ignoring dates
                        if ignore_date or event["date_from"] <= date <= event["date_to"]:
                            article = {
                                "title": title,
                                "content": content,
                                "outlet": "BBC News",
                                "date": date,
                                "event": event["name"],
                                "url": url
                            }
                            articles.append(article)
                            if debug:
                                print(f"[BBC DEBUG] SUCCESS! Added article #{len(articles)}: '{title[:40]}...'")
                            else:
                                print(f"Added BBC article #{len(articles)}: {title[:40]}...")
                        elif debug:
                            print(f"[BBC DEBUG] Article date {date} outside event range {event['date_from']} to {event['date_to']}")
                            articles_outside_date += 1
                    except Exception as e:
                        if debug:
                            print(f"[BBC DEBUG] Date parsing error: {e}")
                            if ignore_date:
                                print("[BBC DEBUG] Using current date since date filtering is disabled")
                                # Create article with current date as fallback
                                article = {
                                    "title": title,
                                    "content": content,
                                    "outlet": "BBC News",
                                    "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "event": event["name"],
                                    "url": url
                                }
                                articles.append(article)
                                print(f"[BBC DEBUG] SUCCESS! Added article #{len(articles)}: '{title[:40]}...'")
                            else:
                                articles_with_missing_data += 1
                                continue
                
                except requests.exceptions.Timeout:
                    if debug:
                        print(f"[BBC DEBUG] Timeout while fetching article {url}, trying next article")
                    continue
                except requests.exceptions.RequestException as e:
                    if debug:
                        print(f"[BBC DEBUG] Request error while fetching article: {e}")
                    continue
                except Exception as e:
                    if debug:
                        print(f"[BBC DEBUG] Error processing article: {e}")
                    continue
                
                if len(articles) >= max_articles:
                    if debug:
                        print(f"[BBC DEBUG] Reached {max_articles} articles, stopping")
                    break
            
            params["page"] += 1
            
            if debug:
                print(f"[BBC DEBUG] Moving to page {params['page']}, collected {len(articles)} articles so far")
                print(f"[BBC DEBUG] Stats: {processed_links} links processed, {articles_outside_date} outside date range, {articles_with_missing_data} missing data")
            
            # Add delay between pages
            time.sleep(2)
            
        except requests.exceptions.Timeout:
            attempts += 1
            if debug:
                print(f"[BBC DEBUG] Timeout on page {params['page']}, attempt {attempts}/{max_attempts}")
            time.sleep(5 * attempts)  # Increasingly longer delay
            
        except Exception as e:
            if debug:
                print(f"[BBC DEBUG] Error scraping BBC: {e}")
            else:
                print(f"Error scraping BBC: {e}")
            attempts += 1
            time.sleep(5)
    
    if debug:
        print(f"[BBC DEBUG] Scraping complete for '{keywords}'. Collected {len(articles)} articles total")
        print(f"[BBC DEBUG] Final stats: Processed {processed_links} links, {articles_outside_date} outside date range, {articles_with_missing_data} missing data")
    
    return articles