import logging
import logging.config
import os

level = os.environ.get('LOG_LEVEL', 'INFO').upper()

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'stream': 'ext://sys.stdout'
        }
    },
    'loggers': {
        'eyepop': {
            'level': level,
            'handlers': ['console'],
            'propagate': False
        },
        'eyepop.requests': {
            'level': 'DEBUG' if level == 'DEBUG' else 'WARNING',
            'handlers': ['console'],
            'propagate': False
        },
        'eyepop.metrics': {
            'level': level,
            'handlers': ['console'],
            'propagate': False
        },
        'eyepop.tracer': {
            'level': level,
            'handlers': ['console'],
            'propagate': False
        },
        'eyepop.compute': {
            'level': level,
            'handlers': ['console'],
            'propagate': False
        }
    }
})
