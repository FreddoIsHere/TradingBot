import sys
import time

from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
import argparse
import re

from utils import force_click, Panel, Mode, script_click_xpath


class Trading212:

    def __init__(self, username, password, headless, mode, long_sleep, short_sleep, timeout):
        # Creating headless browser if required
        self.mode = mode
        self.long_sleep = long_sleep
        self.short_sleep = short_sleep
        self.timeout = timeout

        if headless:
            options = webdriver.ChromeOptions()
            options.headless = True
            self.driver = webdriver.Chrome(executable_path='/home/frederik/TradingBot/chromedriver', options=options)
        else:
            self.driver = webdriver.Chrome(executable_path='/home/frederik/TradingBot/chromedriver')

        self.driver.get("https://www.trading212.com/en/login")  # Getting a website
        self.setup(username, password)

    def setup(self, username, password):
        # Entering username and password
        self.driver.find_element_by_id("username-real").send_keys(username)
        self.driver.find_element_by_id("pass-real").send_keys(password)
        # Login
        self.driver.find_element_by_class_name("button-login").click()
        # Waiting and opening the user menu to avoid the 'You're using CFD' message.
        wait = WebDriverWait(self.driver, self.timeout).until(expected_conditions.element_to_be_clickable((By.CLASS_NAME, "account-menu-button")))
        try:
            script_click_xpath(self.driver, f"//div[@data-dojo-attach-event='click: close' and @class='close-icon']")
        except:
            pass


class Invest(Trading212):

    def __init__(self, username, password, headless=True, long_sleep=0.5, short_sleep=0.1,
                 timeout=30):
        super().__init__(username, password, headless, Mode.Invest, long_sleep, short_sleep, timeout)

    def get_current_positions(self):
        time.sleep(2)
        script_click_xpath(self.driver, f"//span[@data-dojo-attach-event='click: onTabClick' and @class='tab-item tabpositions has-tooltip svg-icon-holder']")
        time.sleep(2)
        list = self.driver.find_elements_by_xpath(
            f"//td[@data-column-id='name' and @class='name']/parent::tr"
        )
        current_positions = []
        for l in list:
            current_positions.append(re.split("_|[a-z]", l.get_attribute('data-code'))[0])
        self.driver.quit()
        return current_positions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='API')
    parser.add_argument('--username', nargs="?", type=str, default='fredkelbel.fk@gmail.com', help='Username')
    parser.add_argument('--password', nargs="?", type=str, default='password', help='Password')
    parser.add_argument('--head', nargs="?", type=bool, default=False, help='Panel')
    args = parser.parse_args()
    print("Retrieving current positions")
    invest = Invest(username=args.username, password=args.password, headless=not args.head)
    print(invest.get_current_positions())
