import logging


def create_and_configure_logger(log_name='log_file.log', level=logging.DEBUG):
    """
    Sets up a logger that works across files.
    The logger prints to console, and to log_name log file.

    Example usage:
        In main function:
            logger = create_and_configer_logger(log_name='myLog.log')

        Then in all other files:
            logger = logging.getLogger(__name__)

        To add records to log:
            logger.debug(f"New Log Message. Value of x is {x}")

    Args:
        log_name: str, log file name

    Returns: logger
    """
    # set up logging to file
    logging.basicConfig(
        filename=log_name,
        level=level,
        format='\n' + '[%(asctime)s - %(levelname)s] {%(pathname)s:%(lineno)d} -' + '\n' + ' %(message)s' + '\n',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(level=level)
    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logger = logging.getLogger(__name__)
    return logger
