import argparse
from stats import *
from api import Invest

parser = argparse.ArgumentParser(description='Run')
parser.add_argument('--username', nargs="?", type=str, default='fredkelbel.fk@gmail.com', help='Username')
parser.add_argument('--password', nargs="?", type=str, default='password', help='Password')
parser.add_argument('--panel', nargs="?", type=str, default='Practice', help='Panel')
parser.add_argument('--mode', nargs="?", type=str, default='Invest', help='Mode')
parser.add_argument('--head', nargs="?", type=bool, default=True, help='Panel')
args = parser.parse_args()
