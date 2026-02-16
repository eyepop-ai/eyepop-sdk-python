import logging
import logging.config
import os

# Libraries should be silent by default. Only configure console output
# if the user explicitly opts in via EYEPOP_LOG_LEVEL or LOG_LEVEL.
_explicit_level = os.environ.get('EYEPOP_LOG_LEVEL', os.environ.get('LOG_LEVEL', '')).upper()

# Always add a NullHandler to prevent "No handlers could be found" warnings
logging.getLogger('eyepop').addHandler(logging.NullHandler())

if _explicit_level:
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
                'level': _explicit_level,
                'handlers': ['console'],
                'propagate': False
            },
            'eyepop.requests': {
                'level': 'DEBUG' if _explicit_level == 'DEBUG' else 'WARNING',
                'handlers': ['console'],
                'propagate': False
            },
            'eyepop.metrics': {
                'level': _explicit_level,
                'handlers': ['console'],
                'propagate': False
            },
            'eyepop.tracer': {
                'level': _explicit_level,
                'handlers': ['console'],
                'propagate': False
            },
            'eyepop.compute': {
                'level': _explicit_level,
                'handlers': ['console'],
                'propagate': False
            }
        }
    })
