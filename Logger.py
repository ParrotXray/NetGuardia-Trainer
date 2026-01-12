import os
import logging
import coloredlogs
from termcolor import colored
from logging.handlers import RotatingFileHandler


class Logger:
    """
    Logger class with optional custom color support.
    """

    def __init__(self, name):
        """
        Initialize a logger instance.
        """
        self.logger = logging.getLogger(name)
        self.level = logging.INFO
        self.datefmt = "%Y-%m-%d %H:%M:%S"
        self.fmt = "%(asctime)s  %(levelname)-5s ThreadId(%(thread)d) %(name)s %(filename)s:%(lineno)d: %(message)s"

        self.logger.setLevel(self.level)

        if not self.logger.handlers:
            if not os.path.exists("logs"):
                os.makedirs("logs", exist_ok=True)

            self.file_handler = RotatingFileHandler(
                os.path.join("logs", "bot.log"),
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )

            coloredlogs.DEFAULT_FIELD_STYLES = {
                "asctime": {"color": "magenta", "bold": True},
                "levelname": {"color": "black", "bold": True},
                "thread": {"color": "blue", "bold": True},
                "name": {"color": "yellow", "bold": True},
                "filename": {"color": "black", "bold": True},
                "lineno": {"color": "black", "bold": True},
            }

            coloredlogs.DEFAULT_LEVEL_STYLES = {
                "debug": {"color": "cyan"},
                "info": {"color": "white"},
                "warning": {"color": "yellow"},
                "error": {"color": "red", "bold": True},
                "critical": {"color": "red"},
            }

            coloredlogs.install(
                level=self.level,
                fmt=self.fmt,
            )

            self.file_handler.setLevel(self.level)
            self.file_handler.setFormatter(
                logging.Formatter(self.fmt, datefmt=self.datefmt)
            )

            self.logger.addHandler(self.file_handler)

    def format_message(self, message, color):
        """
        Customize log messages based on the provided colors.
        """
        if color:
            return colored(message, color=color)
        return message

    def debug(self, message, color=None):
        """
        Log a debug message, optionally with a custom color.
        """
        self.logger.debug(self.format_message(message, color), stacklevel=2)

    def info(self, message, color=None):
        """
        Log an info message, optionally with a custom color.
        """
        self.logger.info(self.format_message(message, color), stacklevel=2)

    def warning(self, message, color=None):
        """
        Log a warning message, optionally with a custom color.
        """
        self.logger.warning(self.format_message(message, color), stacklevel=2)

    def error(self, message, color=None):
        """
        Log an error message, optionally with a custom color.
        """
        self.logger.error(self.format_message(message, color), stacklevel=2)

    def critical(self, message, color=None):
        """
        Log a critical message, optionally with a custom color.
        """
        self.logger.critical(self.format_message(message, color), stacklevel=2)
