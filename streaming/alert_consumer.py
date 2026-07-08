import json
import logging
from datetime import datetime, timezone

from confluent_kafka import Consumer

from streaming.db import get_connection, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
IN_TOPIC = "classified-results"

# only these two tiers are worth waking someone up for
ALERT_TIERS = {"High", "Critical"}


def build_reason(event):
    category = event.get("category", "unknown")
    flags = event.get("compliance_flags") or []
    reason = f"{event['risk_tier']} risk -- {category}"
    if flags:
        reason += f" ({', '.join(flags)})"
    return reason


def raise_alert(event):
    file_hash = event["file_hash"]
    reason = build_reason(event)

    conn = get_connection()
    # INSERT OR IGNORE leans on the unique index on file_hash -- if this file
    # already has an alert (message redelivered), the insert is a no-op.
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO alerts
        (file_hash, file_name, risk_tier, reason, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            file_hash,
            event["file_name"],
            event["risk_tier"],
            reason,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    inserted = cur.rowcount == 1
    conn.close()
    return inserted, reason


def handle_message(event):
    required_fields = {"file_hash", "file_name", "risk_tier"}
    missing = required_fields - event.keys()
    if missing:
        logger.warning(f"skipping a message that's missing {missing}, looks like old/junk data")
        return

    risk_tier = event["risk_tier"]

    # most files aren't alert-worthy -- just let them pass
    if risk_tier not in ALERT_TIERS:
        logger.info(f"{event['file_name']} is {risk_tier}, no alert needed")
        return

    inserted, reason = raise_alert(event)
    if inserted:
        logger.warning(f"ALERT  {event['file_name']}  ->  {reason}")
    else:
        logger.info(f"already alerted on {event['file_name']}, skipping")


def main():
    init_db()

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": "alerting-service",
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([IN_TOPIC])

    logger.info(f"listening on {IN_TOPIC} for high/critical results... (Ctrl+C to stop)")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error(f"consumer error: {msg.error()}")
                continue

            event = json.loads(msg.value().decode("utf-8"))
            handle_message(event)

    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()