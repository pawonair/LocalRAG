"""
Webhook Support
Send notifications on document upload and processing completion.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import json
import threading
import queue
import requests


class WebhookEvent(Enum):
    """Webhook event types."""
    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_PROCESSED = "document.processed"
    DOCUMENT_DELETED = "document.deleted"
    COLLECTION_CREATED = "collection.created"
    COLLECTION_DELETED = "collection.deleted"
    QUERY_COMPLETED = "query.completed"
    ERROR = "error"


@dataclass
class WebhookConfig:
    """Configuration for a webhook endpoint."""
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    retry_count: int = 3
    timeout: float = 10.0


@dataclass
class WebhookPayload:
    """Payload sent to webhook endpoints."""
    event: WebhookEvent
    timestamp: datetime
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class WebhookManager:
    """
    Manager for webhook notifications.
    """

    def __init__(self, async_delivery: bool = True):
        """
        Initialize webhook manager.

        Args:
            async_delivery: Whether to deliver webhooks asynchronously
        """
        self._webhooks: List[WebhookConfig] = []
        self._async_delivery = async_delivery
        self._queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._callbacks: Dict[WebhookEvent, List[Callable]] = {}

        if async_delivery:
            self._start_worker()

    def _start_worker(self):
        """Start async delivery worker thread."""
        if self._worker_thread is not None:
            return

        def worker():
            while True:
                try:
                    item = self._queue.get(timeout=1.0)
                    if item is None:
                        break
                    webhook, payload = item
                    self._deliver(webhook, payload)
                    self._queue.task_done()
                except queue.Empty:
                    continue

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def register(self, config: WebhookConfig) -> None:
        """Register a webhook endpoint."""
        self._webhooks.append(config)

    def unregister(self, url: str) -> bool:
        """Unregister a webhook by URL."""
        for i, webhook in enumerate(self._webhooks):
            if webhook.url == url:
                self._webhooks.pop(i)
                return True
        return False

    def register_callback(
        self,
        event: WebhookEvent,
        callback: Callable[[WebhookPayload], None]
    ) -> None:
        """Register a local callback for an event."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def emit(self, event: WebhookEvent, data: Dict[str, Any]) -> None:
        """
        Emit an event to all registered webhooks.

        Args:
            event: Event type
            data: Event data
        """
        payload = WebhookPayload(
            event=event,
            timestamp=datetime.now(),
            data=data
        )

        # Call local callbacks
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(payload)
                except Exception:
                    pass

        # Deliver to webhooks
        for webhook in self._webhooks:
            if not webhook.enabled:
                continue

            if event not in webhook.events:
                continue

            if self._async_delivery:
                self._queue.put((webhook, payload))
            else:
                self._deliver(webhook, payload)

    def _deliver(self, webhook: WebhookConfig, payload: WebhookPayload) -> bool:
        """
        Deliver payload to webhook endpoint.

        Returns:
            True if successful
        """
        headers = {
            "Content-Type": "application/json",
            **webhook.headers
        }

        if webhook.secret:
            import hashlib
            import hmac
            signature = hmac.new(
                webhook.secret.encode(),
                payload.to_json().encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = signature

        for attempt in range(webhook.retry_count):
            try:
                response = requests.post(
                    webhook.url,
                    data=payload.to_json(),
                    headers=headers,
                    timeout=webhook.timeout
                )

                if response.status_code < 400:
                    return True

            except requests.RequestException:
                pass

        return False

    def list_webhooks(self) -> List[WebhookConfig]:
        """List all registered webhooks."""
        return self._webhooks.copy()

    def clear(self) -> None:
        """Clear all webhooks."""
        self._webhooks.clear()

    def shutdown(self) -> None:
        """Shutdown the webhook manager."""
        if self._async_delivery and self._worker_thread:
            self._queue.put(None)
            self._worker_thread.join(timeout=5.0)


# Global webhook manager instance
_webhook_manager: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """Get global webhook manager instance."""
    global _webhook_manager
    if _webhook_manager is None:
        _webhook_manager = WebhookManager()
    return _webhook_manager


def emit_event(event: WebhookEvent, data: Dict[str, Any]) -> None:
    """Emit an event using the global manager."""
    manager = get_webhook_manager()
    manager.emit(event, data)


# Convenience functions for common events

def emit_document_uploaded(
    document_id: str,
    filename: str,
    file_type: str,
    collection_id: str
) -> None:
    """Emit document uploaded event."""
    emit_event(WebhookEvent.DOCUMENT_UPLOADED, {
        "document_id": document_id,
        "filename": filename,
        "file_type": file_type,
        "collection_id": collection_id
    })


def emit_document_processed(
    document_id: str,
    filename: str,
    chunk_count: int,
    collection_id: str
) -> None:
    """Emit document processed event."""
    emit_event(WebhookEvent.DOCUMENT_PROCESSED, {
        "document_id": document_id,
        "filename": filename,
        "chunk_count": chunk_count,
        "collection_id": collection_id
    })


def emit_document_deleted(document_id: str, filename: str) -> None:
    """Emit document deleted event."""
    emit_event(WebhookEvent.DOCUMENT_DELETED, {
        "document_id": document_id,
        "filename": filename
    })


def emit_query_completed(
    query: str,
    collection_id: str,
    num_results: int,
    processing_time: float
) -> None:
    """Emit query completed event."""
    emit_event(WebhookEvent.QUERY_COMPLETED, {
        "query": query,
        "collection_id": collection_id,
        "num_results": num_results,
        "processing_time": processing_time
    })
