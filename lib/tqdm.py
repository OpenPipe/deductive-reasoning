import sys

try:
    sys.modules["IPython"].get_ipython
    from tqdm import notebook as tqdm
except:
    from tqdm import std as tqdm
