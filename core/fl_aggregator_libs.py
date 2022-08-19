# package for aggregator
from pathlib import Path

from fllibs import *

logDir = os.path.join(os.environ['HOME'], "models", args.model, args.time_stamp, 'aggregator')
time_stamp = args.time_stamp
logDir = os.path.join(args.log_path, "models", args.model, time_stamp, 'aggregator')
if not os.path.isdir(logDir):
    Path(logDir).mkdir(exist_ok=True, parents=True)
logFile = os.path.join(logDir, 'log')

def init_logging():
    logging.basicConfig(
                    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='(%m-%d) %H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(logFile, mode='a'),
                        logging.StreamHandler()
                    ])

def dump_ps_ip():
    hostname_map = {}
    with open('ipmapping', 'rb') as fin:
        hostname_map = pickle.load(fin)

    ps_ip = str(hostname_map[str(socket.gethostname())])
    args.ps_ip = ps_ip

    with open(os.path.join(logDir, 'ip'), 'wb') as fout:
        pickle.dump(ps_ip, fout)

    logging.info(f"Load aggregator ip: {ps_ip}")


def initiate_aggregator_setting():
    init_logging()

initiate_aggregator_setting()
