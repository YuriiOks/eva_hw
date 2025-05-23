# scripts/prepare_deployment.py
import os, json; from datetime import datetime
def create_pkg(): d = f"sim_deploy_{datetime.now().strftime('%y%m%d')}"; os.makedirs(d, exist_ok=True); print(f"Sim deploy pkg: {d}"); return d
def main(): create_pkg()
if __name__ == "__main__": pass
