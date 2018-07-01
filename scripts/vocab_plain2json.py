import os
import sys
import time
import json
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", type=str,
                    help="""Vocabulary in plain text format, whose each line is a word.""")
parser.add_argument("-o", "--output", type=str, default=None,
                    help="""Output file.""")

def INFO(string):
    time_format = '%Y-%m-%d %H:%M:%S'
    sys.stderr.write('{0}: {1}\n'.format(time.strftime(time_format), string))

def main(args):

    worddict = OrderedDict()
    INFO("Begin...")
    with open(args.input) as f:
        for ii, line in enumerate(f):
            worddict[line.strip()] = (ii, 0)

    INFO("Done.")

    if args.output is None:
        args.output = os.path.basename(args.input)

    INFO('Save to {0}'.format('%s.json' % args.output))

    with open('%s.json' % args.output, "w") as f:
        json.dump(worddict, f, indent=1)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

