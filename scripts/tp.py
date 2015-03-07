#!/usr/bin/env python
'''
Created on Mar 5, 2015

@author: Daniel
'''
import sys
from taxi_plate import commands

if __name__ == '__main__':

    usage = """
 Usage: taxi_plate.py <subcommand> <configuration_file>

 Available subcommands:
    """
    usage += "\n    ".join(commands.modes)

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    cmd = sys.argv[1]

    try:
        main = getattr(commands, cmd)
    except AttributeError:
        print(usage)
        sys.exit(1)

    main(sys.argv[2:])

    pass
