"""
Test BETA A/B testing flow with PA establishment trigger.
"""
from playwright.sync_api import sync_playwright, expect
import time

print("=" * 80)
print("TESTING BETA A/B FLOW WITH PA ESTABLISHMENT")
print("=" * 80)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    print("\n📍 Step 1: Navigate to BETA")
    page.goto("http://localhost:8504")
    time.sleep(3)

    # Click BETA tab
    print("  → Clicking BETA tab...")
    beta_tab = page.get_by_text("BETA", exact=True).first
    beta_tab.click()
    time.sleep(2)

    page.screenshot(path=".playwright-mcp/beta_tab_clicked.png")
    print("  ✓ Screenshot: beta_tab_clicked.png")

    print("\n📍 Step 2: Check for Beta Consent Screen")
    time.sleep(2)

    # Look for consent button
    consent_buttons = page.get_by_text("I Consent to Beta Testing")
    if consent_buttons.count() > 0:
        print("  → Found consent screen, clicking consent...")
        consent_buttons.first.click()
        time.sleep(2)
        page.screenshot(path=".playwright-mcp/beta_consented.png")
        print("  ✓ Screenshot: beta_consented.png")
    else:
        print("  → Already consented (or no consent screen)")

    print("\n📍 Step 3: Start Conversation to Establish PA")

    # Find chat input
    chat_input = page.locator('textarea[placeholder*="Ask"]').or_(
        page.locator('textarea[placeholder*="message"]')
    ).or_(
        page.locator('textarea').first
    )

    # Message 1: Establish PA with clear purpose
    print("\n  → Turn 1: Sending message to establish PA...")
    message_1 = "I want to learn about healthy meal planning for a vegetarian diet focused on high protein."

    chat_input.fill(message_1)
    time.sleep(1)

    # Find and click send
    send_button = page.get_by_role("button").filter(has_text="Send").or_(
        page.locator('button[kind="primary"]')
    ).first

    send_button.click()
    print(f"  ✓ Sent: {message_1[:50]}...")

    # Wait for response
    print("  ⏳ Waiting for response...")
    time.sleep(8)

    page.screenshot(path=".playwright-mcp/turn_1_response.png")
    print("  ✓ Screenshot: turn_1_response.png")

    # Message 2: Follow-up to solidify PA
    print("\n  → Turn 2: Follow-up message...")
    message_2 = "What are some good protein sources for vegetarians?"

    chat_input.fill(message_2)
    time.sleep(1)
    send_button.click()
    print(f"  ✓ Sent: {message_2[:50]}...")

    print("  ⏳ Waiting for response (PA should be established now)...")
    time.sleep(8)

    page.screenshot(path=".playwright-mcp/turn_2_response_pa_established.png")
    print("  ✓ Screenshot: turn_2_response_pa_established.png")

    print("\n📍 Step 4: Check for Observation Deck (PA indicator)")

    # Try to open Observation Deck if available
    show_deck_buttons = page.get_by_text("Show Observation Deck")
    if show_deck_buttons.count() > 0:
        print("  → Opening Observation Deck...")
        show_deck_buttons.first.click()
        time.sleep(2)
        page.screenshot(path=".playwright-mcp/observation_deck_pa.png")
        print("  ✓ Screenshot: observation_deck_pa.png")

        # Check for PA content
        page_content = page.content()
        if "Purpose" in page_content or "Primacy Attractor" in page_content:
            print("  ✅ PA content visible in Observation Deck!")
        else:
            print("  ⚠️  No PA content found yet")

    # Message 3: This should trigger A/B testing if PA is established
    print("\n  → Turn 3: Next message (A/B testing should activate)...")
    message_3 = "Can you suggest a week-long meal plan?"

    chat_input.fill(message_3)
    time.sleep(1)
    send_button.click()
    print(f"  ✓ Sent: {message_3[:50]}...")

    print("  ⏳ Waiting for response (A/B testing should be active)...")
    time.sleep(8)

    page.screenshot(path=".playwright-mcp/turn_3_ab_testing.png")
    print("  ✓ Screenshot: turn_3_ab_testing.png")

    print("\n📍 Step 5: Check Browser Console for A/B Testing Logs")

    # Get console logs
    console_logs = []
    page.on("console", lambda msg: console_logs.append(msg.text))

    time.sleep(2)

    print("\n" + "=" * 80)
    print("VISUAL INSPECTION NEEDED:")
    print("=" * 80)
    print("\nCheck these screenshots:")
    print("  1. beta_consented.png - Should show BETA interface")
    print("  2. turn_2_response_pa_established.png - PA should be established by now")
    print("  3. turn_3_ab_testing.png - A/B testing should be active")
    print("  4. observation_deck_pa.png - Should show PA content")

    print("\nNext: Check Supabase for new deltas!")
    print("  → governance_deltas table")
    print("  → Filter: mode = 'beta'")
    print("  → Look for: test_condition, shown_response_source, baseline_fidelity")

    print("\n⏸  Browser will stay open for 30 seconds for inspection...")
    time.sleep(30)

    browser.close()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
