import os
import subprocess
import py7zr
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("data", 'iris', "Dataset name ['iris', 'al', 'pb']")
flags.DEFINE_string("dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iris_data'), "Directory path where to load data")
flags.DEFINE_boolean("prpr", True, "Whether to download preprocessed data")

dois = {
  'iris': ('10.5281/zenodo.7524572','unknown'), #key name: (preprocessed doi link, raw doi link)
  'al': ('10.5281/zenodo.7674274','unknown'),
  'pb': ('10.5281/zenodo.7674366','10.5281/zenodo.5724362')
  }

def main():
  # download data
  if FLAGS.prpr:
    result = subprocess.run(['zenodo_get'], ['-d'], [dois[FLAGS.data][0]], ['-o'], [FLAGS.dir])
  else:
    assert dois[FLAGS.data][1]!='unknown', "Unknown data address, download manually"
    result = subprocess.run(['zenodo_get'], ['-d'], [dois[FLAGS.data][1]], ['-o'], [FLAGS.dir])
    with py7zr.SevenZipFile("data_raw.7z", 'r') as archive:
      archive.extractall(path=FLAGS.dir)
    os.remove(os.path.join(write_dir, "data_longformat.7z"))

if __name__ == '__main__':
  app.run(main)
