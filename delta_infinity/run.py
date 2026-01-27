# run.py
from delta_infinity.core import DeltaInfinity

if __name__ == "__main__":
    config_path = r"C:\github\CS-Peirce\delta_infinity\config.yaml"   # path to config
    delta = DeltaInfinity(config_path)
    delta.run()