import json
import logging
from datetime import datetime, timezone

from confluent_kafka import Consumer, Producer

from discovery.ingestion.file_ingester import FileIngester
from discovery.scanner.pattern_scanner import PatternScanner
from discovery.classifier.cascade_classifier import CascadeClassifier
from streaming.db import get_connection, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
IN_TOPIC = "raw-documents"
OUT_TOPIC = "classified-results"

ingester = FileIngester()
scanner = PatternScanner()
classifier = CascadeClassifier()


def already_classified(file_hash):
    conn = get_connection()
    row = conn.execute(
        "SELECT 1 FROM classified_files WHERE file_hash = ?", (file_hash,)
    ).fetchone()
    conn.close()
    return row is not None


def save_result(event, result):
    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO classified_files
        (file_hash, file_path, file_name, category, risk_tier, decided_by_tier,
         confidence, compliance_flags, notes, detected_at, classified_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event["file_hash"],
            event["file_path"],
            event["file_name"],
            result.category,
            result.risk_tier,
            result.decided_by_tier,
            result.confidence,
            json.dumps(result.compliance_flags),
            result.notes,
            event["detected_at"],
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def handle_message(event, producer):
    required_fields = {"file_hash", "file_path", "file_name", "detected_at"}
    missing = required_fields - event.keys()
    if missing:
        logger.warning(f"skipping a message that's missing {missing}, looks like old/junk data")
        return

    file_hash = event["file_hash"]

    # same file might come through twice (Kafka only promises at-least-once),
    # so just bail out early if we've already got a result for this hash
    if already_classified(file_hash):
        logger.info(f"already have a result for {event['file_name']}, skipping")
        return

    ingested = ingester.ingest(event["file_path"])
    if ingested["error"]:
        logger.warning(f"couldn't read {event['file_name']}: {ingested['error']}")
        return

    text = ingested["text"] or ""
    findings = scanner.scan(text)
    result = classifier.classify(text, findings)

    save_result(event, result)
    logger.info(
        f"{event['file_name']} -> {result.category} "
        f"({result.risk_tier}, tier {result.decided_by_tier}, conf {result.confidence})"
    )

    summary = {
        "file_hash": file_hash,
        "file_name": event["file_name"],
        "category": result.category,
        "risk_tier": result.risk_tier,
        "decided_by_tier": result.decided_by_tier,
        "confidence": result.confidence,
        "compliance_flags": result.compliance_flags,
        "classified_at": datetime.now(timezone.utc).isoformat(),
    }
    producer.produce(
        topic=OUT_TOPIC,
        key=file_hash,
        value=json.dumps(summary).encode("utf-8"),
    )
    producer.poll(0)


def main():
    init_db()

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": "classification-service",
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([IN_TOPIC])

    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})

    logger.info(f"listening on {IN_TOPIC}... (Ctrl+C to stop)")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error(f"consumer error: {msg.error()}")
                continue

            event = json.loads(msg.value().decode("utf-8"))
            handle_message(event, producer)

    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    main()