import sys
from app import app, predict_grade, ScorePayload

# 10 Skenario Uji Sesuai ML Deploy Plan
TEST_SCENARIOS = [
    {"listening": 0, "structure": 0, "reading": 0, "expected": "beginner", "combo": "B, B, B"},
    {"listening": 20, "structure": 20, "reading": 50, "expected": "beginner", "combo": "B, B, I"},
    {"listening": 20, "structure": 20, "reading": 90, "expected": "intermediate", "combo": "B, B, A"},
    {"listening": 20, "structure": 50, "reading": 90, "expected": "intermediate", "combo": "B, I, A"},
    {"listening": 24, "structure": 84, "reading": 90, "expected": "intermediate", "combo": "B, A, A"},
    {"listening": 50, "structure": 50, "reading": 50, "expected": "intermediate", "combo": "I, I, I"},
    {"listening": 60, "structure": 70, "reading": 86, "expected": "advanced", "combo": "I, I, A"},
    {"listening": 60, "structure": 90, "reading": 90, "expected": "advanced", "combo": "I, A, A"},
    {"listening": 100, "structure": 100, "reading": 100, "expected": "advanced", "combo": "A, A, A"},
    {"listening": 90, "structure": 20, "reading": 50, "expected": "intermediate", "combo": "A, B, I"},
]

def run_tests():
    print("=" * 60)
    print("Memulai Verifikasi Inferensi Model IAET-SIM XGBoost")
    print("=" * 60)

    passed = 0
    failed = 0

    for idx, tc in enumerate(TEST_SCENARIOS, 1):
        payload = ScorePayload(
            listening_score=tc["listening"],
            structure_score=tc["structure"],
            reading_score=tc["reading"],
        )
        res = predict_grade(payload)
        actual = res["grade"]
        sub = res["sub_levels"]
        combo_actual = f"{sub['listening'][0]}, {sub['structure'][0]}, {sub['reading'][0]}"

        if actual == tc["expected"]:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = f"❌ FAIL (Expected: {tc['expected']})"

        print(f"[{idx:02d}] {status} | Skor: L={tc['listening']:3d} S={tc['structure']:3d} R={tc['reading']:3d} | Level: {combo_actual} | Grade: {actual.upper()}")

    print("=" * 60)
    print(f"Hasil: {passed}/{len(TEST_SCENARIOS)} Skenario Berhasil Lulus.")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()

