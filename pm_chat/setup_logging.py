import logging

def setup_logging():
    # Configure logging to show the DEBUG output for requests
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('requests').setLevel(logging.DEBUG)


    logger = logging.getLogger(__name__)
    logger.info('Logging is setup...')
    return logger

logger = setup_logging()
