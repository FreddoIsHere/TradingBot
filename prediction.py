import argparse
from model import Model
from api import Invest
import time
import sys

parser = argparse.ArgumentParser(description='Run')
parser.add_argument('--username', nargs="?", type=str, default='fredkelbel.fk@gmail.com', help='Username')
parser.add_argument('--password', nargs="?", type=str, help='Password')
parser.add_argument('--head', nargs="?", type=bool, default=False, help='Open browser')
parser.add_argument('--ticker', nargs="?", type=str, help='Ticker')
args = parser.parse_args()
if args.ticker is None:
    print("Retrieving current positions...")
    invest = Invest(username=args.username, password=args.password, headless=not args.head)
    tickers = invest.get_current_positions()
else:
    tickers = [args.ticker]
model = Model()
for t in tickers:
    start_time = time.time()
    try:
        prediction, confidence = model.predict_signal(t)
        print("Today's recommendation for {} is: {} with a confidence of {}%".format(t, prediction, confidence))
    except:
        print("No information on {} could be retrieved!".format(t))
    elapsed_time = time.time()
    time_to_sleep = int(13 - (elapsed_time - start_time))
    for i in range(time_to_sleep, 0, -1):  # only 5 api calls per minute allowed
        sys.stdout.write("\r")
        sys.stdout.write("Waiting time for next API call: {:2d}s".format(i))
        sys.stdout.flush()
        time.sleep(1)