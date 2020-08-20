import argparse
from stats import *
from api import Invest

parser = argparse.ArgumentParser(description='Map Generator')
parser.add_argument('--username', nargs="?", type=str, default='fredkelbel.fk@gmail.com', help='Username')
parser.add_argument('--password', nargs="?", type=str, default='password', help='Password')
parser.add_argument('--panel', nargs="?", type=str, default='Practice', help='Panel')
parser.add_argument('--mode', nargs="?", type=str, default='Invest', help='Mode')
parser.add_argument('--head', nargs="?", type=bool, default=True, help='Panel')
args = parser.parse_args()
#invest = Invest(username=args.username, password=args.password, panel=args.panel, headless=not args.head)
#invest.buy_stock(stock='GOOGL', amount=1)
print(get_moving_average('MSFT'))
