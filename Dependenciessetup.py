import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = ['tensorflow','python-binance','gym','tf-agents','numpy','pandas']
for package in packages:
    install(package)