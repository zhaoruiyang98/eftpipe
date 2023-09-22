from cobaya.mpi import root_only
from cobaya.log import logger_setup


@root_only
def mpi_info(logger, msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)


@root_only
def mpi_warning(logger, msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)
