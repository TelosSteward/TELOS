# JB Protocols

**Purpose**: Document JB's preferences, working style, and communication protocols for Claude Code interactions.

**Status**: Living document - will be expanded over time and eventually converted to a Claude Code skill.

---

## Communication & Decision Making

### Protocol 1: Multi-Select Options (Added: 2025-10-30)

**Context**: When presenting questions or options to JB for decision making.

**Preference**:
- ✅ **ALWAYS use multi-select** when presenting options/choices
- ✅ **ALWAYS include "All of the above" option** for every question
- ❌ **NEVER force single-choice** when multiple options might apply
- ❌ **NEVER make JB copy/paste or manually type out multiple selections**

**Rationale**: JB often wants to select multiple options simultaneously. Forcing single-choice creates unnecessary friction and requires manual workarounds.

**Implementation**:
```python
# Example - CORRECT approach:
AskUserQuestion(
    questions=[{
        "question": "What criteria should we use?",
        "multiSelect": true,  # ← ALWAYS TRUE
        "options": [
            {"label": "Option A", "description": "..."},
            {"label": "Option B", "description": "..."},
            {"label": "Option C", "description": "..."},
            {"label": "All of the above", "description": "Select all criteria"}  # ← ALWAYS INCLUDE
        ]
    }]
)
```

**Quote from JB**:
> "I don't like not being able to select multiple choices when you provide me options moving forward sometimes there are several that apply and I don't want to have to copy and paste what I want or write it out."

> "Its basically my primacy Attractor for claude code. haha"

---

## Future Protocols

_This section will be populated with additional preferences as they emerge from conversations._

### Protocol Template

**Protocol N: [Name]** (Added: YYYY-MM-DD)

**Context**: [When this applies]

**Preference**:
- [Specific preferences]

**Rationale**: [Why JB prefers this]

**Implementation**: [How to implement]

---

**Last Updated**: 2025-10-30
**Maintained by**: Claude Code sessions
**Future**: Will be converted to Claude Code skill for automatic application
