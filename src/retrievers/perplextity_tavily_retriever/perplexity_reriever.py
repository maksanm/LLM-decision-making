from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from seleniumbase import Driver
from bs4 import BeautifulSoup
import os
import time

class PerplexityRetriever:

    def __init__(self, domains=None):
        self.domains = domains


    def invoke(self, input):
        perplexity_prompt = input
        if self.domains:
            perplexity_prompt += f". Focus on these sources: '{"', '".join(self.domains)}'."
        return self._retrieve_from_perplexity(perplexity_prompt)


    def _retrieve_from_perplexity(self, perplexity_prompt):
        # Configure Selenium WebDriver using Chrome in GUI mode with Undetected ChromeDriver enabled
        perplexity_driver = Driver(uc=True, headless=False, multi_proxy=True)
        # Open URL using UC mode with 3 second reconnect time to bypass initial detection
        perplexity_driver.uc_open_with_reconnect(os.getenv("PERPLEXITY_URL"), reconnect_time=3)
        soup = BeautifulSoup(perplexity_driver.page_source, 'html.parser')
        # Find the textarea element using BeautifulSoup
        textarea = soup.find('textarea')
        # If the textarea is found, use Selenium to interact with it
        if textarea:
            # Find the textarea element using Selenium
            input_box = perplexity_driver.find_element(By.TAG_NAME, 'textarea')
            # Enter the prompt
            input_box.send_keys(perplexity_prompt)
            # Submit the prompt by pressing Enter
            input_box.send_keys(Keys.RETURN)
        # Wait for the response to be fully loaded
        wait = WebDriverWait(perplexity_driver, 60)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.prose span")))
        time.sleep(10)
        # Get the updated page source
        page_source = perplexity_driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        # Find the div containing the response
        response_div = soup.find('div', class_='prose')
        # Extract the text from all spans within the div
        response_text = ''.join(span.get_text() for span in response_div.find_all('span'))

        perplexity_driver.quit()
        return response_text

