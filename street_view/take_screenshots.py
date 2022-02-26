import argparse
import os
from datetime import datetime
from shutil import rmtree
from time import sleep

from pynput.keyboard import Listener, Key
from selenium.common.exceptions import ElementNotInteractableException
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from seleniumwire import webdriver


def is_page_loaded(driver: webdriver.Chrome):
    print('###############')

    for r in driver.requests[-4:]:
        if r.response:
            print(r.response.status_code, r.host, r.response.headers['Content-Type'])

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

    print('Set the position of the streetview and press a key')
    input()
    print('Press esc to stop the image acquisition')

    e = None
    while True:
        try:
            e = driver.find_element(By.TAG_NAME, 'canvas')
            break
        except ElementNotInteractableException:
            continue

    while Listener(on_press=lambda key: key != Key.esc):
        e.send_keys(Keys.UP)
        sleep(1)

        while not is_page_loaded(driver):
            sleep(1)

        driver.save_screenshot(os.path.join(out_folder, f"{datetime.now().isoformat()}.png"))
        sleep(1)


if __name__ == '__main__':
    main()
