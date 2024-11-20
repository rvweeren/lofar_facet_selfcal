import sys
import os

# Add the parent directory to sys.path to allow imports from it
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)  # Insert at the start to prioritize this path

# Import the main function from facetselfcal.py in the parent directory
from ..facetselfcal import main as facetselfcal_main

def main():
    # Run the main function from facetselfcal
    facetselfcal_main()

if __name__ == "__main__":
    main()
