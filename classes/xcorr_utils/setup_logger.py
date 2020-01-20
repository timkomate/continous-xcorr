import logging
from datetime import datetime

now = datetime.now()
dt_string = now.strftime("%Y-%m-%d-%H:%M:%S")

log_format = "%(asctime)s::%(levelname)s::%(filename)s::%(lineno)d::%(message)s"
logging.basicConfig(
    filename='./logs/%s.log' % (dt_string), 
    level='INFO', 
    format=log_format
)

logger = logging.getLogger("cross-correlation")
