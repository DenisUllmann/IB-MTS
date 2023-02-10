import os
import subprocess
import py7zr
import gdown
from absl import app
from absl import flags

FLAGS = flags.FLAGS

doi_testres = ''

root_dir = os.path.dirname(os.path.abspath(__file__))

def main():
  # download data
    temp = os.path.join(root_dir, "testres.7z")
    result = subprocess.run(['zenodo_get'], ['-d'], [doi_testres], ['-o'], [root_dir])
    with py7zr.SevenZipFile(temp, 'r') as archive:
      archive.extractall(path=root_dir)
    os.remove(temp)

if __name__ == '__main__':
  app.run(main)
