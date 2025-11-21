"""
Test script to verify BETA mode conversation interface fix.

This test validates that after completing the BETA intro,
the conversation interface appears correctly and users can send messages.
"""

from playwright.sync_api import sync_playwright
import time
import sys

def test_beta_conversation_flow():
    """Test the complete DEMO -> BETA -> conversation flow"""

    print("🧪 Starting BETA mode conversation flow test...")

    with sync_playwright() as p:
        # Launch browser with visible UI for debugging
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # Navigate to the application
        print("📍 Navigating to application...")
        page.goto("http://localhost:8501")
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        # Check if DEMO mode is active by default
        print("✅ Checking initial DEMO mode...")
        demo_button = page.locator("button:has-text('DEMO')")
        assert demo_button.is_visible(), "DEMO tab button should be visible"

        # Complete DEMO mode to unlock BETA
        print("🎯 Completing DEMO mode...")
        # Navigate through DEMO slides by pressing Enter multiple times
        for i in range(15):  # 15 slides total (0-14)
            page.keyboard.press("Enter")
            time.sleep(0.5)

        # After DEMO completion, send a test message to confirm DEMO works
        demo_input = page.locator("textarea[placeholder='Ask Steward anything about TELOS...']")
        if demo_input.is_visible():
            print("✅ DEMO mode completed - input field visible")
            # Don't actually send message to avoid API calls

        # Click BETA tab
        print("🔄 Switching to BETA mode...")
        beta_button = page.locator("button:has-text('BETA')")
        beta_button.click()
        time.sleep(2)

        # Verify BETA intro appears
        print("📋 Checking BETA intro slides...")
        welcome_text = page.locator("text=/Welcome to the BETA Experience/i")
        assert welcome_text.is_visible(), "BETA intro should appear"

        # Navigate through BETA intro slides
        print("📍 Navigating through BETA intro...")
        for slide in range(3):  # 3 slides before final
            next_button = page.locator("button:has-text('Next')")
            if next_button.is_visible():
                next_button.click()
                time.sleep(1)

        # Click "Start Beta Testing" button
        print("🚀 Starting BETA testing...")
        start_button = page.locator("button:has-text('Start Beta Testing')")
        assert start_button.is_visible(), "Start Beta Testing button should be visible"
        start_button.click()
        time.sleep(2)

        # CRITICAL TEST: Check if input field appears after beta intro
        print("🔍 Checking for conversation interface...")

        # Look for the input field - should be visible now
        input_field = page.locator("textarea[placeholder='Tell TELOS']").first

        # Also check for form element
        form_element = page.locator("form").first

        if input_field.is_visible():
            print("✅ SUCCESS: Input field is visible after BETA intro!")
            print("✅ Users can now send messages in BETA mode")

            # Test entering text (but don't send to avoid API calls)
            input_field.fill("Test message in BETA mode")
            print("✅ Text entry works in input field")

            # Check Send button is present
            send_button = page.locator("button:has-text('Send')")
            assert send_button.is_visible(), "Send button should be visible"
            print("✅ Send button is visible")

            return True
        else:
            print("❌ FAILURE: Input field NOT visible after BETA intro!")
            print("❌ Issue: Conversation interface did not appear")

            # Debug info
            print("\n🔍 Debug Information:")
            print(f"Form element visible: {form_element.is_visible() if form_element else 'Not found'}")

            # Take screenshot for debugging
            page.screenshot(path="beta_mode_error.png")
            print("📸 Screenshot saved as beta_mode_error.png")

            return False

        browser.close()

def test_session_state():
    """Test session state values after BETA intro completion"""

    print("\n🔍 Testing session state consistency...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # Navigate to app
        page.goto("http://localhost:8501")
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        # Complete DEMO and switch to BETA
        for i in range(15):
            page.keyboard.press("Enter")
            time.sleep(0.3)

        beta_button = page.locator("button:has-text('BETA')")
        beta_button.click()
        time.sleep(1)

        # Complete BETA intro
        for _ in range(3):
            next_button = page.locator("button:has-text('Next')")
            if next_button.is_visible():
                next_button.click()
                time.sleep(0.5)

        start_button = page.locator("button:has-text('Start Beta Testing')")
        start_button.click()
        time.sleep(2)

        # Check page state through console
        result = page.evaluate("""
            () => {
                // Try to access Streamlit session state
                const stateInfo = {
                    hasInput: !!document.querySelector('textarea[placeholder="Tell TELOS"]'),
                    hasForm: !!document.querySelector('form'),
                    hasSendButton: !!document.querySelector('button:has-text("Send")')
                };
                return stateInfo;
            }
        """)

        print(f"✅ Input field present: {result['hasInput']}")
        print(f"✅ Form element present: {result['hasForm']}")
        print(f"✅ Send button present: {result['hasSendButton']}")

        browser.close()

        return all([result['hasInput'], result['hasForm'], result['hasSendButton']])

if __name__ == "__main__":
    print("=" * 60)
    print("BETA MODE CONVERSATION INTERFACE FIX TEST")
    print("=" * 60)

    # Make sure Streamlit app is running
    print("\n⚠️  Please ensure the Streamlit app is running on http://localhost:8501")
    print("Run: streamlit run telos_observatory_v3/main.py")
    input("\nPress Enter when ready to start testing...")

    # Run tests
    test1_passed = test_beta_conversation_flow()
    test2_passed = test_session_state()

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED!")
        print("✅ BETA mode conversation interface is working correctly")
        print("✅ Users can now interact with TELOS after completing BETA intro")
    else:
        print("❌ Some tests failed. Please review the output above.")
        sys.exit(1)