import logging
import logging.config
import os
from typing import Optional


def get_logging_config(level: Optional[str] = None) -> dict:
    """Get logging configuration dict for the EyePop SDK.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
               Defaults to LOG_LEVEL environment variable or INFO.

    Returns:
        Dictionary suitable for logging.config.dictConfig()
    """
    if level is None:
        level = os.environ.get('EYEPOP_LOG_LEVEL', 'INFO')

    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
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
                'level': level.upper(),
                'handlers': ['console'],
                'propagate': False
            },
            'eyepop.requests': {
                'level': 'DEBUG' if level.upper() == 'DEBUG' else 'WARNING',
                'handlers': ['console'],
                'propagate': False
            },
            'eyepop.metrics': {
                'level': level.upper(),
                'handlers': ['console'],
                'propagate': False
            },
            'eyepop.tracer': {
                'level': level.upper(),
                'handlers': ['console'],
                'propagate': False
            },
            'eyepop.compute': {
                'level': level.upper(),
                'handlers': ['console'],
                'propagate': False
            }
        }
    }


def configure_logging(level: Optional[str] = None, config: Optional[dict] = None) -> None:
    """
    Configure EyePop SDK logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
               Defaults to LOG_LEVEL environment variable or INFO.
        config: Optional custom logging config dict. If provided, it will be used
                instead of the default config. Must follow logging.config.dictConfig format.

    Examples:
        Basic usage:
        >>> from eyepop.logging import configure_logging
        >>> configure_logging(level='DEBUG')

        Custom config with different formatter for requests:
        >>> config = get_logging_config('INFO')
        >>> config['formatters']['requests_fmt'] = {'format': '%(levelname)s - %(message)s'}
        >>> config['handlers']['requests_handler'] = {
        ...     'class': 'logging.StreamHandler',
        ...     'formatter': 'requests_fmt'
        ... }
        >>> config['loggers']['eyepop.requests']['handlers'] = ['requests_handler']
        >>> configure_logging(config=config)
    """
    if config is None:
        config = get_logging_config(level)

    logging.config.dictConfig(config)
