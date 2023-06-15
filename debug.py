from interface import app
import sys, signal
import warnings
warnings.filterwarnings('ignore')

"""
NOTE: not needed with WSGI server, this is for debug server only
"""
def signal_handler(sig, frame):
    """
    Bypass keyboard interrupt shenanigang
    """
    sys.exit(0)
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    port = sys.argv[1]

    app.config['DEBUG'] = False
    app.run(host="0.0.0.0", port=port, debug=True)
