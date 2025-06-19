# run_cli.py
import argparse, json
from mini_flow import run

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--text", required=True, help="Email body to classify")
args = parser.parse_args()

print(json.dumps(run(args.text), indent=2))
