"""Kiem tra moi truong chay collector."""
import sys

errors = []
warnings = []

def check(name, fn):
    try:
        result = fn()
        print(f"  [OK] {name:<30} {result}")
    except Exception as e:
        errors.append(name)
        print(f"  [FAIL] {name:<28} LỖITL: {e}")

def check_warn(name, fn):
    try:
        result = fn()
        print(f"  [OK] {name:<30} {result}")
    except Exception as e:
        warnings.append(name)
        print(f"  [WARN] {name:<28} Thieu: {e}")

print()
print("=" * 60)
print("  KIEM TRA MOI TRUONG CARLA COLLECTOR")
print("=" * 60)

print(f"\nPython: {sys.version}")
print(f"Path:   {sys.executable}\n")

print("[1] Core packages ---")
check("carla",          lambda: __import__("carla").__file__)
check("numpy",          lambda: __import__("numpy").__version__)
check("Pillow (PIL)",   lambda: __import__("PIL").__version__)
check("opencv-python",  lambda: __import__("cv2").__version__)

print("\n[2] Standard library ---")
check("csv",            lambda: "OK")
check("json",           lambda: "OK")
check("math",           lambda: "OK")
check("pathlib",        lambda: "OK")
check("queue",          lambda: "OK")
check("threading",      lambda: "OK")

print("\n[3] Optional packages ---")
check_warn("matplotlib",    lambda: __import__("matplotlib").__version__)
check_warn("pandas",        lambda: __import__("pandas").__version__)
check_warn("torch",         lambda: __import__("torch").__version__)

print("\n[4] CARLA API check ---")
try:
    import carla
    client_class = carla.Client
    check("carla.Client",       lambda: str(client_class))
    check("carla.LaneType",     lambda: str(carla.LaneType.Driving))
    check("carla.TrafficLightState", lambda: str(carla.TrafficLightState.Red))
except Exception as e:
    errors.append("carla API")
    print(f"  [FAIL] carla API: {e}")

print()
print("=" * 60)
if errors:
    print(f"  X  {len(errors)} LOI: {', '.join(errors)}")
    print("  => CHUA SAN SANG chay collector!")
else:
    print("  OK  TAT CA DIEU KIEN CHAY COLLECTOR DA DAP UNG")
if warnings:
    print(f"  !  {len(warnings)} canh bao (khong bat buoc): {', '.join(warnings)}")
print("=" * 60)
print()

sys.exit(1 if errors else 0)
