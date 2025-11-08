from src.core.logging_config import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("Hello from adaptive-reasoning-agent!")


if __name__ == "__main__":
    main()
