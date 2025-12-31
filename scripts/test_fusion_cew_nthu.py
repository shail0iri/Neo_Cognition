import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.fusion_cew_nthu import FusionCEWNTHU

print("🧠 Testing Fusion Engine (CEW + NTHU)")

try:
    engine = FusionCEWNTHU()

    # Define test scenarios
    test_scenarios = [
        {
            "name": "ALERT DRIVER",
            "eye_img": np.ones((80, 80), dtype=np.uint8) * 200,  # Bright = open eyes
            "features": {
                "participant_id": 1,
                "frame_id": 100,
                "left_ear": 0.35,
                "right_ear": 0.34,
                "avg_ear": 0.345,
                "blink": 0,
                "ear_asymmetry": 0.01,
            }
        },
        {
            "name": "DROWSY DRIVER",
            "eye_img": np.ones((80, 80), dtype=np.uint8) * 50,   # Dark = closed eyes
            "features": {
                "participant_id": 1,
                "frame_id": 200,
                "left_ear": 0.15,
                "right_ear": 0.16,
                "avg_ear": 0.155,
                "blink": 1,
                "ear_asymmetry": 0.05,
            }
        }
    ]

    for scenario in test_scenarios:
        print(f"\n📊 Testing: {scenario['name']}")

        result = engine.predict_from_eye_and_features(
            scenario["eye_img"],
            scenario["features"]
        )

        print("=== FUSION OUTPUT ===")
        for k, v in result.items():
            print(f"{k}: {v}")

    print("\n✅ Fusion engine test completed successfully!")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()