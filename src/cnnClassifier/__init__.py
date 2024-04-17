import os, sys, logging

loggingStr = '[%(asctime)s: %(levelname)s: %(module)s: %(message)s]'

logDir = 'logs'
logFilePath = os.path.join(logDir, 'running_logs.log')
os.makedirs(logDir, exist_ok=True)

logging.basicConfig(
  level=logging.INFO,
  format=loggingStr,
  handlers=[
    logging.FileHandler(logFilePath),
    logging.StreamHandler(sys.stdout)
  ]
)

logger = logging.getLogger('cnnClassifierLogger')