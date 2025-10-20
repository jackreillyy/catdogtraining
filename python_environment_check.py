import importlib, sys
def _v(m):
    try:
        mod = importlib.import_module(m)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "MISSING"
def check_packages():
    pkgs = ["torch","torchvision","numpy","matplotlib","PIL"]
    print("Python:", sys.version)
    for p in pkgs:
        print(f"{p}:", _v("PIL" if p=="PIL" else p))
    missing = [p for p in ["torch","torchvision"] if _v(p)=="MISSING"]
    if missing:
        raise ImportError("Missing required packages: " + ", ".join(missing))
if __name__ == "__main__":
    check_packages()
