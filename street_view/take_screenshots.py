import argparse
import os
from datetime import datetime
from shutil import rmtree
from time import sleep

from selenium.common.exceptions import ElementNotInteractableException, NoSuchWindowException
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from seleniumwire import webdriver


def is_page_loaded(driver: webdriver.Chrome):
    latest_response = []
    while len(latest_response) != 2:
        for r in driver.requests[-4:]:
            if r.response and r.response.status_code == 204 and r.host == 'www.google.com':
                latest_response.append(r)

            if len(latest_response) == 2:
                break

    r1, r2 = latest_response

    if (not r1.response or not r2.response) or (r1.response.status_code != 204 and r2.response.status_code != 204):
        return False

    for r in [r1.response.headers['Content-Type'], r2.response.headers['Content-Type']]:
        if r not in ['image/gif', 'text/html; charset=UTF-8']:
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Take screenshot from Google Street View')
    parser.add_argument('-o', '--output', type=str, help='Output folder (es. output)', required=True)
    args = parser.parse_args()
    out_folder = args.output

    if os.path.exists(out_folder):
        rmtree(out_folder)

    os.mkdir(out_folder)

    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    driver.get("https://www.google.com/maps")

    while driver.title != 'Google Maps':
        sleep(1)

    print('Set the position of the streetview and press a key. '
          'Then close the browser to stop screenshot acquisition.')
    input()

    e = None
    while True:
        try:
            e = driver.find_element(By.TAG_NAME, 'canvas')
            break
        except ElementNotInteractableException:
            continue

    while True:
        try:
            e.send_keys(Keys.UP)
            sleep(1)

            while not is_page_loaded(driver):
                sleep(1)

            driver.save_screenshot(os.path.join(out_folder, f"{datetime.now().isoformat()}.png"))
            sleep(1)
        except NoSuchWindowException:
            break

    print(f'Screenshot stored in {out_folder} folder')


if __name__ == '__main__':
    main()
