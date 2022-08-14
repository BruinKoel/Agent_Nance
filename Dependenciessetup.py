import subprocess
import sys


packages = ['tensorflow','python-binance','gym','tf-agents','numpy','pandas']


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for package in packages:
    install(package)