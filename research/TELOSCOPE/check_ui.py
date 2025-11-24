#!/usr/bin/env python3
"""Check the UI with Playwright"""
from playwright.sync_api import sync_playwright
import time

def check_ui():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the app
        page.goto('http://localhost:8502')
        time.sleep(3)

        # Take a screenshot
        page.screenshot(path='ui_check_screenshot.png', full_page=True)
        print("✅ Screenshot saved to ui_check_screenshot.png")

        # Check if borders are visible by inspecting computed styles
        demo_button = page.locator('button:has-text("DEMO")').first
        if demo_button.is_visible():
            print("✅ DEMO button is visible")

            # Get computed style
            border_color = demo_button.evaluate('el => getComputedStyle(el).borderColor')
            border_width = demo_button.evaluate('el => getComputedStyle(el).borderWidth')
            print(f"  Border color: {border_color}")
            print(f"  Border width: {border_width}")

        browser.close()

if __name__ == "__main__":
    check_ui()
