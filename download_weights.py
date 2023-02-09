import os
import subprocess
import py7zr
import gdown
from absl import app
from absl import flags

FLAGS = flags.FLAGS

vgg_weights_adress = 'https://drive.google.com/uc?id=1HOzmKQFljTdKWftEP-kWD7p2paEaeHM0'
ts_predmodel_weights = '10.5281/zenodo.7568871'
class_iris_weights = ''

addresses = [
vgg_weights_adress,
ts_predmodel_weights,
class_iris_weights
]

types_dwnl = [
'goog_drive',
'zenodo_doi',
'zenodo_doi'
]

root_dir = os.path.dirname(os.path.abspath(__file__))
write_dirs = [
os.path.join(root_dir,'vgg_weights'),
root_dir,
os.path.join(root_dir,'classifiers')
]

def main():
  # download data
  for address, type_dwnl, write_dir in zip(addresses, types_dwnl, write_dir):
    if type_dwnl == 'zenodo_doi':
      result = subprocess.run(['zenodo_get'], ['-d'], [address], ['-o'], [write_dir])
      with py7zr.SevenZipFile(os.path.join(write_dir, "weights.7z"), 'r') as archive:
        archive.extractall(path=write_dir)
      os.remove(os.path.join(write_dir, "weights.7z"))
    if type_dwnl == 'goog_drive':
      gdown.download(address, write_dir, quiet=False)

if __name__ == '__main__':
  app.run(main)
