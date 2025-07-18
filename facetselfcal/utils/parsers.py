import re
import os

def parse_source_id(inp_str: str = None):
    """
    Parse ILTJhhmmss.ss±ddmmss.s source_id string

    Args:
        inp_str: ILTJ source_id

    Returns: ILT‐coordinate string

    """

    try:
        parsed_inp = re.findall(r'ILTJ\d{6}\.\d{2}[+\-]\d{6}\.\d{1}', inp_str)[0]
    except IndexError:
        print(f"WARNING: {inp_str} does not contain a valid source ID")
        parsed_inp = ''

    return parsed_inp


def parse_history(ms, hist_item):
    """
    Grep specific history item from MS
    :param ms: measurement set
    :param hist_item: history item
    :return: parsed string
    """
    hist = os.popen('taql "SELECT * FROM ' + ms + '::HISTORY" | grep ' + hist_item).read().split(' ')
    for item in hist:
        if hist_item in item and len(hist_item) <= len(item):
            return item
    print('WARNING:' + hist_item + ' not found')
    return None