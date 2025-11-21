#!/usr/bin/env python3
"""Quick test to check bottom navigation buttons."""

import asyncio
from playwright.async_api import async_playwright
from pathlib import Path
from datetime import datetime

async def quick_test():
    screenshot_dir = Path(".playwright-mcp")
    screenshot_dir.mkdir(exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()

        print("Quick test - checking bottom nav buttons...")

        # Navigate and accept consent
        await page.goto("http://localhost:8504")
        await page.wait_for_timeout(2000)

        # Accept consent
        consent_checkbox = page.locator('input[type="checkbox"]')
        if await consent_checkbox.count() > 0:
            await consent_checkbox.click()
            await page.wait_for_timeout(500)
            continue_button = page.locator('button:has-text("Continue")')
            await continue_button.click()
            await page.wait_for_timeout(2000)

        # Go to DEMO
        demo_tab = page.locator('button:has-text("DEMO")')
        await demo_tab.first.click()
        await page.wait_for_timeout(2000)

        # Start demo
        start = page.locator('button:has-text("Start Demo")')
        await start.click()
        await page.wait_for_timeout(2000)

        # Advance to slide 4
        for _ in range(4):
            next_btn = page.locator('button:has-text("Next")').first
            await next_btn.click()
            await page.wait_for_timeout(1000)

        # Open observation deck
        show_deck = page.locator('button:has-text("Show Observation Deck")')
        await show_deck.first.click()
        await page.wait_for_timeout(2000)

        # Scroll to bottom
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1000)

        # Take screenshot
        screenshot = screenshot_dir / f"bottom_nav_test_{datetime.now().strftime('%H%M%S')}.png"
        await page.screenshot(path=str(screenshot), full_page=True)
        print(f"Screenshot: {screenshot}")

        # Check buttons
        prev_btn = page.locator('button:has-text("⬅️ Previous")')
        next_btn = page.locator('button:has-text("Next ➡️")')
        hide_btn = page.locator('button:has-text("Hide Observation Deck")')

        print(f"Previous buttons found: {await prev_btn.count()}")
        print(f"Next buttons found: {await next_btn.count()}")
        print(f"Hide buttons found: {await hide_btn.count()}")

        # Try to get the visible ones
        print("\nChecking visibility...")
        for i in range(await prev_btn.count()):
            is_visible = await prev_btn.nth(i).is_visible()
            print(f"  Previous button #{i}: visible={is_visible}")

        for i in range(await next_btn.count()):
            is_visible = await next_btn.nth(i).is_visible()
            print(f"  Next button #{i}: visible={is_visible}")

        print("\nKeeping browser open for 30 seconds...")
        await page.wait_for_timeout(30000)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(quick_test())
