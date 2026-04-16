import json
import os

COST_FILE = "data/cost.json"

def track_cost(response, is_embedding=False):
    if not hasattr(response, "usage") or not response.usage:
        return
        
    usage = response.usage
    try:
        if is_embedding:
            # $0.02 / 1M tokens
            cost = usage.total_tokens * 0.02 / 1_000_000
        else:
            # $0.15 / 1M prompt, $0.60 / 1M completion
            cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000
            
        if os.path.exists(COST_FILE):
            with open(COST_FILE) as f:
                data = json.load(f)
        else:
            data = {"total": 0.0}
            
        data["total"] += cost
        with open(COST_FILE, "w") as f:
            json.dump(data, f)
            
    except Exception as e:
        print(f"Error tracking cost: {e}")

def get_total_cost() -> float:
    if os.path.exists(COST_FILE):
        try:
            with open(COST_FILE) as f:
                data = json.load(f)
                return float(data.get("total", 0.0))
        except:
            return 0.0
    return 0.0
