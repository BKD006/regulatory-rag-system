import os
import logging
from datetime import datetime
import structlog
from dotenv import load_dotenv
load_dotenv()

class CustomLogger:

    def __init__(self, log_dir="logs"):
        # mode: dev | prod
        self.log_mode = os.getenv("LOG_MODE").lower()
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        # create logs dir only in dev
        if self.log_mode != "prod":
            os.makedirs(self.logs_dir, exist_ok=True)
        # timestamped file only for dev
        if self.log_mode != "prod":
            log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
            self.log_file_path = os.path.join(self.logs_dir,log_file)
        else:
            self.log_file_path = None

    def get_logger(self, name=__file__):
        logger_name = os.path.basename(name)

        # prevent duplicate handlers (important for ECS / uvicorn reload)
        root_logger = logging.getLogger()
        if root_logger.handlers:
            return structlog.get_logger(logger_name)
        log_level = os.getenv("LOG_LEVEL").upper()
        handlers = []

        # --------------------------------------------------
        # Console handler (always needed for ECS)
        # --------------------------------------------------

        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(console_handler)

        # --------------------------------------------------
        # File handler only in dev
        # --------------------------------------------------

        if self.log_mode != "prod" and self.log_file_path:
            file_handler = logging.FileHandler(self.log_file_path)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            handlers.append(file_handler)

        # --------------------------------------------------
        # basic logging config
        # --------------------------------------------------

        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            handlers=handlers,
            force=True,  # important for uvicorn / ECS
        )

        # --------------------------------------------------
        # structlog config
        # --------------------------------------------------

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso",utc=True,key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger(logger_name)