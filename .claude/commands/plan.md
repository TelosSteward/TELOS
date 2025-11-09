Initialize structured planning mode using the StructuredPlanner tool.

When this command is run:

1. Import the StructuredPlanner from scripts/structured_planning.py
2. Ask the user what they want to plan
3. Create a comprehensive plan with phases, steps, risks, and success criteria
4. Generate both JSON and Markdown outputs
5. Present the plan for user approval

Use this Python code to initialize:

```python
import sys
sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_CLEAN')
from scripts.structured_planning import StructuredPlanner

# Get project name from user
planner = StructuredPlanner("Project Name Here")

# Build plan systematically
planner.add_phase(...)
planner.add_risk(...)
planner.add_success_criterion(...)

# Output results
planner.print_summary()
planner.save_plan()
planner.generate_markdown_report()
```
