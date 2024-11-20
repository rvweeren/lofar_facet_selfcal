import sys
import os

# Add the parent directory to sys.path to allow imports from it
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

print(sys.path)

# Import the main function from facetselfcal.py in the parent directory
from facetselfcal import main as facetselfcal_main

def main():
    # Run the main function from facetselfcal
    facetselfcal_main()

if __name__ == "__main__":
    main()
