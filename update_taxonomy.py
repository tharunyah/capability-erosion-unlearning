# update_taxonomy.py
import json
from collections import Counter

NEW_SAFETY_CRITICAL_CLASSES = [3, 31, 43, 88, 97, 79]  # bear, elephant, lion, tiger, wolf, spider

with open("data/capability_taxonomy.json", "r") as f:
    taxonomy = json.load(f)

for cls in NEW_SAFETY_CRITICAL_CLASSES:
    cls_str = str(cls)
    current_tier = taxonomy.get(cls_str)
    if current_tier != "long_tail":
        print(f"WARNING: class {cls} is currently '{current_tier}', not 'long_tail' — check before proceeding")
    taxonomy[cls_str] = "safety_critical"

with open("data/capability_taxonomy.json", "w") as f:
    json.dump(taxonomy, f, indent=4)

print("Updated tier counts:")
print(Counter(taxonomy.values()))
