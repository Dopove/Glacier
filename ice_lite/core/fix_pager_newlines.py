import re
from pathlib import Path

file_path = Path("/mnt/d2/Projects/Glacier/Glacier_Release_Staging/ice_lite/core/pager.py")

with open(file_path, "r") as f:
    content = f.read()

# Pattern to find literal newlines within f-strings or multiline strings
# This is a bit tricky with regex, so we'll do it in steps.
# First, identify potential multi-line f-strings/strings
# Then, replace literal newlines with '\n' within those identified blocks.

# This approach is more robust: find string literals that contain actual newlines
# and then replace those newlines with '\n'.

# A simple regex to catch string literals potentially spanning lines
# This is a heuristic and might not catch all cases, but should cover the common ones.
# We'll look for strings starting with f" or " and ending with " that contain newlines.

# Replace newlines within triple-quoted strings
content = re.sub(r'("""[^"]*""")', lambda m: m.group(0).replace('\n', '\\n'), content, flags=re.DOTALL)
content = re.sub(r"('''[^']*''')", lambda m: m.group(0).replace('\n', '\\n'), content, flags=re.DOTALL)

# For f-strings like f"..." that contain newlines within them
# This is harder to do with a single regex, so let's try to target the specific patterns identified
# from the traceback: f"{content}\n\n[ENGINE]" and "reasoning_text = "\n".join" etc.

# Fix the specific pattern identified at line 232 (now fixed by previous replace)
# No, actually the content was already modified from the first replace!
# I need to get the original content again.
# The user committed some changes, let's read it from the file again to get the latest.
