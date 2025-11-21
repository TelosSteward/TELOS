#!/usr/bin/env python3
"""
Playwright test for Observation Deck navigation improvements.
Tests the new Previous/Next buttons at the bottom of Observation Deck.
"""

import asyncio
from playwright.async_api import async_playwright
from pathlib import Path
from datetime import datetime

async def test_observation_deck_navigation():
    """Test Observation Deck bottom navigation buttons."""

    screenshot_dir = Path(".playwright-mcp")
    screenshot_dir.mkdir(exist_ok=True)

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()

        print("🔭 Testing Observation Deck Navigation...")

        # Test 1: Load beta page
        print("\n📸 Test 1: Loading DEMO tab")
        await page.goto("http://localhost:8504")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(2000)

        # Accept consent
        try:
            consent_checkbox = page.locator('input[type="checkbox"]')
            if await consent_checkbox.count() > 0:
                await consent_checkbox.click()
                await page.wait_for_timeout(500)

                continue_button = page.locator('button:has-text("Continue to Beta")')
                if await continue_button.count() > 0:
                    await continue_button.click()
                    await page.wait_for_timeout(2000)
                    print("   ✓ Consent accepted")
        except Exception as e:
            print(f"   ⚠ Consent step: {e}")

        # Click DEMO tab
        try:
            demo_tab = page.locator('button:has-text("DEMO")')
            if await demo_tab.count() > 0:
                await demo_tab.first.click()
                await page.wait_for_timeout(2000)
                print("   ✓ DEMO tab active")
        except Exception as e:
            print(f"   ⚠ DEMO tab: {e}")

        # Test 2: Start demo
        print("\n📸 Test 2: Starting Demo")
        try:
            start_demo = page.locator('button:has-text("Start Demo")')
            if await start_demo.count() > 0:
                await start_demo.click()
                await page.wait_for_timeout(2000)
                print("   ✓ Demo started")

                screenshot_path = screenshot_dir / f"demo_started_{datetime.now().strftime('%H%M%S')}.png"
                await page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"   ✓ Screenshot: {screenshot_path}")
        except Exception as e:
            print(f"   ⚠ Start demo: {e}")

        # Test 3: Navigate to slide 4-5 where Observation Deck appears
        print("\n📸 Test 3: Navigate to slide with Observation Deck")
        try:
            # Click Next a few times to get to slide 4
            for i in range(4):
                next_button = page.locator('button:has-text("Next")')
                if await next_button.count() > 0:
                    await next_button.first.click()
                    await page.wait_for_timeout(1500)
                    print(f"   ✓ Advanced to slide {i+1}")

            screenshot_path = screenshot_dir / f"slide_4_{datetime.now().strftime('%H%M%S')}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"   ✓ Screenshot: {screenshot_path}")
        except Exception as e:
            print(f"   ⚠ Navigation: {e}")

        # Test 4: Open Observation Deck
        print("\n📸 Test 4: Open Observation Deck")
        try:
            show_deck = page.locator('button:has-text("Show Observation Deck")')
            if await show_deck.count() > 0:
                await show_deck.first.click()
                await page.wait_for_timeout(2000)
                print("   ✓ Observation Deck opened")

                screenshot_path = screenshot_dir / f"obs_deck_open_{datetime.now().strftime('%H%M%S')}.png"
                await page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"   ✓ Screenshot: {screenshot_path}")
        except Exception as e:
            print(f"   ⚠ Open deck: {e}")

        # Test 5: Scroll down to see bottom navigation
        print("\n📸 Test 5: Scroll to bottom navigation")
        try:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1000)

            screenshot_path = screenshot_dir / f"obs_deck_bottom_{datetime.now().strftime('%H%M%S')}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"   ✓ Screenshot: {screenshot_path}")
        except Exception as e:
            print(f"   ⚠ Scroll: {e}")

        # Test 6: Check for Previous/Next buttons at bottom
        print("\n📸 Test 6: Test bottom navigation buttons")
        try:
            # Look for Previous button with obs_deck_prev key
            prev_button = page.locator('button:has-text("Previous")')
            next_button_bottom = page.locator('button:has-text("Next")')
            hide_button = page.locator('button:has-text("Hide Observation Deck")')

            prev_count = await prev_button.count()
            next_count = await next_button_bottom.count()
            hide_count = await hide_button.count()

            print(f"   Previous button: {'✓ Found' if prev_count > 0 else '⚠ Not found'} (count: {prev_count})")
            print(f"   Next button: {'✓ Found' if next_count > 0 else '⚠ Not found'} (count: {next_count})")
            print(f"   Hide button: {'✓ Found' if hide_count > 0 else '⚠ Not found'} (count: {hide_count})")

            # Try clicking Next at bottom
            if next_count > 1:  # Should be 2: one at top, one at bottom
                # Click the bottom one (last one)
                await next_button_bottom.last.click()
                await page.wait_for_timeout(2000)
                print("   ✓ Clicked bottom Next button")

                screenshot_path = screenshot_dir / f"after_bottom_next_{datetime.now().strftime('%H%M%S')}.png"
                await page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"   ✓ Screenshot: {screenshot_path}")
        except Exception as e:
            print(f"   ⚠ Bottom navigation test: {e}")

        # Test 7: Try Previous button
        print("\n📸 Test 7: Test Previous button at bottom")
        try:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1000)

            prev_button = page.locator('button:has-text("Previous")')
            if await prev_button.count() > 1:  # Should be 2: one at top, one at bottom
                await prev_button.last.click()
                await page.wait_for_timeout(2000)
                print("   ✓ Clicked bottom Previous button")

                screenshot_path = screenshot_dir / f"after_bottom_prev_{datetime.now().strftime('%H%M%S')}.png"
                await page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"   ✓ Screenshot: {screenshot_path}")
        except Exception as e:
            print(f"   ⚠ Previous button test: {e}")

        # Test 8: Final state
        print("\n📸 Test 8: Final state capture")
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1000)

        screenshot_path = screenshot_dir / f"final_state_{datetime.now().strftime('%H%M%S')}.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"   ✓ Screenshot: {screenshot_path}")

        print("\n✅ Navigation testing complete! Check .playwright-mcp/ for screenshots.")

        # Keep browser open for inspection
        print("\n⏸  Browser will stay open for 20 seconds for manual inspection...")
        await page.wait_for_timeout(20000)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_observation_deck_navigation())
