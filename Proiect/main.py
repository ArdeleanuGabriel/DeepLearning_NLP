from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


def scrape_google_news(query, from_date, to_date, headless=True):
    # Set up Chrome options
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    # Initialize WebDriver
    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 10)

    driver.get("https://www.google.com/")

    try:
        # Accept Google's consent page if present
        consent_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "I agree")]')))
        consent_button.click()
    except Exception:
        pass  # Consent button may not appear

    # Navigate to Google News
    driver.get("https://www.google.com/search?q=&hl=en&tbm=nws")

    # Wait for search box to be interactable
    search_box = wait.until(EC.element_to_be_clickable((By.NAME, "q")))
    search_query = f'{query} after:{from_date} before:{to_date}'
    search_box.clear()
    search_box.send_keys(search_query + Keys.RETURN)

    time.sleep(3)  # Wait for results to load

    # Extract news articles
    articles = []
    results = driver.find_elements(By.CSS_SELECTOR, "div.dbsr")
    for result in results:
        try:
            title_element = result.find_element(By.CSS_SELECTOR, "div.JheGif.nDgy9d")
            link_element = result.find_element(By.CSS_SELECTOR, "a")
            source_element = result.find_element(By.CSS_SELECTOR, "div.CEMjEf span")

            title = title_element.text
            link = link_element.get_attribute("href")
            source = source_element.text

            articles.append({
                "title": title,
                "link": link,
                "source": source,
            })
        except Exception:
            continue  # Skip if any element is missing

    # Close the browser
    driver.quit()
    return articles


# Example usage
if __name__ == "__main__":
    QUERY = "stock market"
    FROM_DATE = "2024-01-01"
    TO_DATE = "2024-01-31"

    news_articles = scrape_google_news(QUERY, FROM_DATE, TO_DATE)
    for idx, article in enumerate(news_articles, start=1):
        print(f"{idx}. {article['title']}")
        print(f"   Source: {article['source']}")
        print(f"   URL: {article['link']}\n")
