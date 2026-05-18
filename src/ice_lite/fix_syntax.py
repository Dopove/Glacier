import re

with open("test_glacier.py", "r") as f:
    content = f.read()

content = content.replace('print(f"\n--- Running Test: {test_name} ---")', 'print(f"\\n--- Running Test: {test_name} ---")')
content = content.replace('print(f"❌ {test_name}: FAILED\n   Reason: {e}")', 'print(f"❌ {test_name}: FAILED\\n   Reason: {e}")')
content = content.replace('print("\n\n" + "="*50)', 'print("\\n\\n" + "="*50)')
content = content.replace('print(f"\n■ Test: {name}")', 'print(f"\\n■ Test: {name}")')
content = content.replace('print("\n" + "="*50)', 'print("\\n" + "="*50)')

with open("test_glacier.py", "w") as f:
    f.write(content)
