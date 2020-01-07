import logging
import platform
import os
import configs


class Logger(object):

    def __init__(self, path, prefix):
        self.path = path
        self.logging_file = os.path.join(path, prefix + configs.env.paths.logging_file)
        self._mkfile()
        self.logger = self._setlogger()

    def _mkfile(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        open(self.logging_file, 'a') if platform.system() == 'Windows' else \
            (not os.path.exists(self.logging_file) and os.mknod(self.logging_file))

    def _setlogger(self):
        global _utils_logger
        if '_utils_logger' not in globals():
            _utils_logger = logging.getLogger()
        else:
            for idx in reversed(range(len(_utils_logger.handlers))):
                _utils_logger.handlers[idx].close()
                _utils_logger.removeHandler(_utils_logger.handlers[idx])

        _utils_logger.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s :  %(message)s')

        handler = logging.FileHandler(self.logging_file)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        _utils_logger.addHandler(handler)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        _utils_logger.addHandler(console)

        return _utils_logger

    def info(self, msg):
        self.logger.info(msg)
