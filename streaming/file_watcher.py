import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from confluent_kafka import Producer
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WATCHED_DIR = Path("data/raw")
TOPIC = "raw-documents"
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"


SETTLE_DELAY_SECONDS = 1.0


def hash_file(path: Path) -> str:
   
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def delivery_report(err, msg):
   
    if err is not None:
        logger.error(f"Delivery failed for key={msg.key()}: {err}")
    else:
        logger.info(
            f"Delivered to {msg.topic()} [partition {msg.partition()}] "
            f"at offset {msg.offset()}"
        )


class NewFileHandler(FileSystemEventHandler):
    def __init__(self, producer: Producer):
        self.producer = producer

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        time.sleep(SETTLE_DELAY_SECONDS)

        try:
            file_hash = hash_file(path)
        except (FileNotFoundError, PermissionError) as e:
           
            logger.warning(f"Could not hash {path.name} yet, skipping: {e}")
            return

        message = {
            "file_path": str(path),
            "file_name": path.name,
            "file_hash": file_hash,
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }

        self.producer.produce(
            topic=TOPIC,
            key=file_hash,
            value=json.dumps(message).encode("utf-8"),
            callback=delivery_report,
        )
       
        self.producer.poll(0)

        logger.info(f"Published event for {path.name} (hash={file_hash[:12]}...)")


def main():
    WATCHED_DIR.mkdir(parents=True, exist_ok=True)

    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})
    handler = NewFileHandler(producer)

    observer = PollingObserver()
    observer.schedule(handler, str(WATCHED_DIR), recursive=False)
    observer.start()

    logger.info(f"Watching {WATCHED_DIR.resolve()} for new files... (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping file watcher...")
        observer.stop()

    observer.join()
  
    producer.flush()


if __name__ == "__main__":
    main()