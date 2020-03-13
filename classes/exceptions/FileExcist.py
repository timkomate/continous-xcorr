class FileExcistError(Exception):
    def __init__(self, msg):
        print msg