"""
Quick script to check what the BETA mode system actually did for the last turn.
This will help us verify if it's showing Native vs TELOS response.
"""

import streamlit as st
from pathlib import Path

# This won't work as a standalone script, but shows you what to check in session state
print("\n" + "="*80)
print("BETA MODE TURN VERIFICATION")
print("="*80)

print("\nTo verify what response was shown, check these session state variables:")
print("\n1. st.session_state.get('beta_current_turn') - Current turn number")
print("2. st.session_state.get('beta_turn_data') - All turn data including:")
print("   - 'shown_source': Which response was displayed ('telos' or 'native')")
print("   - 'telos_analysis': The TELOS fidelity score and metrics")
print("   - 'native_response': The ungoverned response")
print("   - 'shown_response': What the user actually saw")
print("\n3. Check StateManager turns:")
print("   - st.session_state.state_manager.state.turns[-1]['beta_shown_source']")
print("   - st.session_state.state_manager.state.turns[-1]['fidelity']")

print("\n" + "="*80)
print("WHAT TO LOOK FOR:")
print("="*80)
print("\nIf shown_source == 'native':")
print("  ✅ EXPECTED: Response answers PB&J question")
print("  ✅ EXPECTED: Fidelity shown is still LOW (< 0.3) because it's MEASURED from TELOS")
print("  ⚠️  This is CORRECT A/B test behavior!")
print("\nIf shown_source == 'telos':")
print("  ✅ EXPECTED: Response redirects back to TELOS topics")
print("  ✅ EXPECTED: Fidelity shown is LOW (< 0.3)")
print("  ⚠️  Intervention should be triggered")

print("\n" + "="*80)
print("THE KEY INSIGHT:")
print("="*80)
print("\nThe fidelity score is ALWAYS calculated from TELOS analysis,")
print("even when showing the Native response. This is intentional!")
print("\nBoth responses are generated, both are measured, but only ONE is shown.")
print("The measurement helps populate the Observatory for post-study review.")
print("="*80)
