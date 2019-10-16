import logging
from datetime import datetime

log_format = "%(asctime)s::%(levelname)s::"\
             "%(filename)s::%(lineno)d::%(message)s"

now = datetime.now()

print("now =", now)
dt_string = now.strftime("%Y-%m-%d-%H:%M")
print("date and time =", dt_string)

log_format = "%(asctime)s::%(levelname)s::%(name)s::" \
             "%(filename)s::%(lineno)d::%(message)s"
logging.basicConfig(filename='./logs/%s.log' % (dt_string), level='DEBUG', format=log_format)
logger = logging.getLogger("cross-correlation")
