from sys import exit
from subprocess import run, CalledProcessError

def sva_stack(mslist: list = None, output: str = None):
    """
    Sidereal visibility averaging to speed-up imaging

    Args:
        mslist: MeasurementSet list
        output: Output MeasurementSet name
    """

    cmd = ["sva", "--msout", output] + mslist

    try:
        run(cmd, check=True)
    except CalledProcessError:
        print("sva command not found or failed. Attempting to run from source...")
        run(["git", "clone", "https://github.com/jurjen93/sidereal_visibility_avg", "sva"], check=False)
        try:
            alt_cmd = ["python", "-m", "sva.sidereal_visibility_avg.main", "--msout", output] + mslist
            run(alt_cmd, check=True)
        except CalledProcessError:
            exit("Sidereal visibility averaging failed."
                 "\nVerify that package exists in container:"
                 "\nhttps://github.com/jurjen93/sidereal_visibility_avg")
