import urllib.request
import json
import ssl

BASE_URL = "https://sarvadubey-cicd-repair-env.hf.space"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

def post(endpoint, data):
    req = urllib.request.Request(f"{BASE_URL}{endpoint}", data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, context=CTX, timeout=15) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Error calling {endpoint}: {e}")
        return None

def test_tier(tier_id, actions):
    print(f"\n--- Testing {tier_id} ---")
    obs = post("/reset", {"task_id": tier_id})
    if not obs: return False
    
    for action in actions:
        res = post("/step", {"action_id": action})
        if not res: return False
        info = res.get("info", {})
        print(f"Action {action} -> Increment: {res.get('reward')}, Cumulative: {info.get('cumulative_reward')}")
    
    if res.get("info", {}).get("pipeline_healthy"):
        print(f"[OK] {tier_id} pipeline healthy!")
        return True
    else:
        print(f"[FAILED] {tier_id} pipeline not healthy.")
        return False

success = True
success &= test_tier("tier_1", [1])
success &= test_tier("tier_2", [4, 2])
success &= test_tier("tier_3", [3, 4, 6])

if success:
    print("\nALL REMOTE ENDPOINTS VERIFIED AND WORKING PERFECTLY!")
else:
    print("\nSOME ENDPOINTS FAILED.")
