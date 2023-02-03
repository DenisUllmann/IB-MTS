import os
import subprocess
import py7zr
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("data", 'iris', "Dataset name ['iris', 'al', 'pb']")
flags.DEFINE_string("dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PrPr_data'), "Directory path where to load data")

dois = {
  'iris': '',
  'al': '',
  'pb': ''
  }

def main():
  result = subprocess.run(['zenodo_get'])

if __name__ == '__main__':
  app.run(main)
