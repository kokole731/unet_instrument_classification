import config as cfg
import argparse

parser = argparse.ArgumentParser(description='Source separation trainer')
parser.add_argument('--config', help='Path to config file', required=True)
args = parser.parse_args()

config = cfg.load(args.config)
print(config['depth'])