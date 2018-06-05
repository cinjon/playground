import os
import sys

path_to_server_files = '/var/www/manager'
sys.path.insert(0, path_to_server_files)
activate_this = os.path.join(path_to_server_files, 'venv/bin/activate_this.py')
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))

from app import app as application
