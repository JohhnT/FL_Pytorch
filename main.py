#!/usr/bin/env python3

import argparse
import logging
from server import Server
from utils import config

# Set up parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./configs/BreastCancer/breastcancer.json',
                        help='Federated learning configuration file.')
    parser.add_argument('-l', '--log', type=str, default='INFO',
                        help='Log messages level.')
    parser.add_argument('-d', '--dataset', type=str, default='BreastCancer',
                        help='the name of dataset')
    parser.add_argument('-o', '--out', type=str,
                        default='', help='output file')

    args = parser.parse_args()
    # Set logging
    if args.out == '':
        logging.basicConfig(
            format='[%(levelname)s][%(asctime)s]: %(message)s',
            level=getattr(logging, args.log.upper()),
            datefmt='%H:%M:%S')
    else:
        logging.basicConfig(
            filename=args.out,
            filemode='a',
            format='[%(levelname)s][%(asctime)s]: %(message)s',
            level=getattr(logging, args.log.upper()),
            datefmt='%H:%M:%S')
    logging.info("config:{},  log:{}".format(args.config, args.log))
    # load config
    config = config.Config(args.config)
    server = Server(config)
    server.run()
