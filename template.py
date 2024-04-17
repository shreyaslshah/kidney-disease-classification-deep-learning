import os
from pathlib import Path
import logging

# logging string
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

projectName = 'cnnClassifier'

listOfFiles = [
    '.github/workflows/.gitkeep',
    f'src/{projectName}/__init__.py',
    f'src/{projectName}/components/__init__.py',
    f'src/{projectName}/utils/__init__.py',
    f'src/{projectName}/config/__init__.py',
    f'src/{projectName}/config/__init__.py',
    f'src/{projectName}/pipeline/__init__.py',
    f'src/{projectName}/entity/__init__.py',
    f'src/{projectName}/constants/__init__.py',
    'config/config.yaml',
    'dvc.yaml',
    'params.yaml',
    'requirements.txt',
    'setup.py',
    'research/trials.ipynb',
    'templates/index.html'
]

for filePath in listOfFiles:
  filePath = Path(filePath)
  fileDir, fileName = os.path.split(filePath)

  if fileDir != '':
    os.makedirs(fileDir, exist_ok=True)
    logging.info(f'creating directory: {fileDir} for the file: {fileName}')

  if ((not os.path.exists(filePath)) or (os.path.getsize(filePath) == 0)):
    with open(filePath, 'w') as f:
      pass
    logging.info(f'creating empty file: {filePath}')

  else:
    logging.info(f'{fileName} already exists')
