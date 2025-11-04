import logging
import logging.config

# LOGGING CONFIGURATION

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,

    "formatters": {
        "standard": {"format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"},
        "detailed": {"format": "%(asctime)s | %(levelname)s | %(name)s | %(filename)s | %(lineno)s | %(funcName)s | %(message)s"},
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "data_file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/datalog.log",
            "encoding": "utf-8",
            "mode": "a"
        },
        "preprocessing_file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/preprocessing_log.log",
            "encoding": "utf-8",
            "mode": "a"
        }
    },

    "root": {"level": "INFO", "handlers": ["console"]},

    "loggers": {
        "hkt-mop.data": {
            "level": "DEBUG",
            "handlers": ["data_file"],
            "propagate": True
        },
        "hkt-mop.preprocessing": {
            "level": "DEBUG",
            "handlers": ["preprocessing_file"],
            "propagate": True
        }
    }
}


def setup_logging():
    logging.config.dictConfig(LOGGING)
