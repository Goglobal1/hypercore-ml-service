"""
Alert System Real-time Push - WebSocket and SSE Support
Provides real-time alert notifications to connected dashboards.
"""

from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
import asyncio
import json
import logging
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# MESSAGE TYPES
# =============================================================================

class MessageType(str, Enum):
    """Types of real-time messages."""
    ALERT_FIRED = "alert_fired"
    ALERT_SUPPRESSED = "alert_suppressed"
    STATE_CHANGE = "state_change"
    EPISODE_OPENED = "episode_opened"
    EPISODE_CLOSED = "episode_closed"
    ACKNOWLEDGMENT = "acknowledgment"
    ESCALATION = "escalation"
    SYSTEM_STATUS = "system_status"
    HEARTBEAT = "heartbeat"


@dataclass
class RealtimeMessage:
    """A real-time message to be pushed to clients."""
    message_type: MessageType
    timestamp: datetime
    patient_id: Optional[str]
    risk_domain: Optional[str]
    data: Dict[str, Any]
    priority: int = 0  # 0=normal, 1=high, 2=critical

    def to_json(self) -> str:
        return json.dumps({
            "type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "patient_id": self.patient_id,
            "risk_domain": self.risk_domain,
            "data": self.data,
            "priority": self.priority,
        })

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        return f"event: {self.message_type.value}\ndata: {self.to_json()}\n\n"


# =============================================================================
# CONNECTION MANAGEMENT
# =============================================================================

@dataclass
class ClientConnection:
    """Represents a connected client."""
    connection_id: str
    connected_at: datetime
    client_type: str  # "websocket" or "sse"
    # Subscription filters
    patient_ids: Set[str] = field(default_factory=set)  # Empty = all patients
    risk_domains: Set[str] = field(default_factory=set)  # Empty = all domains
    roles: Set[str] = field(default_factory=set)  # Clinician roles
    # Connection state
    send_callback: Optional[Callable] = None
    last_heartbeat: Optional[datetime] = None
    message_count: int = 0

    def matches_filter(self, patient_id: Optional[str], risk_domain: Optional[str]) -> bool:
        """Check if message matches this client's subscription filters."""
        # If no filters, receive all
        if not self.patient_ids and not self.risk_domains:
            return True

        # Check patient filter
        if self.patient_ids and patient_id and patient_id not in self.patient_ids:
            return False

        # Check domain filter
        if self.risk_domains and risk_domain and risk_domain not in self.risk_domains:
            return False

        return True


class ConnectionManager:
    """Manages WebSocket and SSE connections."""

    def __init__(self):
        self._connections: Dict[str, ClientConnection] = {}
        self._patient_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # patient_id -> connection_ids
        self._domain_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # domain -> connection_ids
        self._role_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # role -> connection_ids
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._broadcast_task: Optional[asyncio.Task] = None

    def register_connection(
        self,
        connection_id: str,
        client_type: str,
        send_callback: Callable,
        patient_ids: Optional[Set[str]] = None,
        risk_domains: Optional[Set[str]] = None,
        roles: Optional[Set[str]] = None,
    ) -> ClientConnection:
        """Register a new connection."""
        connection = ClientConnection(
            connection_id=connection_id,
            connected_at=datetime.now(timezone.utc),
            client_type=client_type,
            patient_ids=patient_ids or set(),
            risk_domains=risk_domains or set(),
            roles=roles or set(),
            send_callback=send_callback,
            last_heartbeat=datetime.now(timezone.utc),
        )

        self._connections[connection_id] = connection

        # Update subscription indices
        for patient_id in connection.patient_ids:
            self._patient_subscriptions[patient_id].add(connection_id)
        for domain in connection.risk_domains:
            self._domain_subscriptions[domain].add(connection_id)
        for role in connection.roles:
            self._role_subscriptions[role].add(connection_id)

        logger.info(f"Registered {client_type} connection: {connection_id}")
        return connection

    def unregister_connection(self, connection_id: str) -> bool:
        """Unregister a connection."""
        if connection_id not in self._connections:
            return False

        connection = self._connections[connection_id]

        # Remove from subscription indices
        for patient_id in connection.patient_ids:
            self._patient_subscriptions[patient_id].discard(connection_id)
        for domain in connection.risk_domains:
            self._domain_subscriptions[domain].discard(connection_id)
        for role in connection.roles:
            self._role_subscriptions[role].discard(connection_id)

        del self._connections[connection_id]
        logger.info(f"Unregistered connection: {connection_id}")
        return True

    def update_subscription(
        self,
        connection_id: str,
        patient_ids: Optional[Set[str]] = None,
        risk_domains: Optional[Set[str]] = None,
        roles: Optional[Set[str]] = None,
    ) -> bool:
        """Update subscription filters for a connection."""
        if connection_id not in self._connections:
            return False

        connection = self._connections[connection_id]

        # Update patient subscriptions
        if patient_ids is not None:
            # Remove old
            for patient_id in connection.patient_ids:
                self._patient_subscriptions[patient_id].discard(connection_id)
            # Add new
            connection.patient_ids = patient_ids
            for patient_id in patient_ids:
                self._patient_subscriptions[patient_id].add(connection_id)

        # Update domain subscriptions
        if risk_domains is not None:
            for domain in connection.risk_domains:
                self._domain_subscriptions[domain].discard(connection_id)
            connection.risk_domains = risk_domains
            for domain in risk_domains:
                self._domain_subscriptions[domain].add(connection_id)

        # Update role subscriptions
        if roles is not None:
            for role in connection.roles:
                self._role_subscriptions[role].discard(connection_id)
            connection.roles = roles
            for role in roles:
                self._role_subscriptions[role].add(connection_id)

        return True

    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self._connections)

    def get_connections_for_message(
        self,
        patient_id: Optional[str],
        risk_domain: Optional[str],
        target_roles: Optional[List[str]] = None,
    ) -> List[ClientConnection]:
        """Get all connections that should receive a message."""
        matching_connections = []

        for conn_id, connection in self._connections.items():
            # Check filters
            if not connection.matches_filter(patient_id, risk_domain):
                continue

            # Check role routing
            if target_roles and connection.roles:
                if not connection.roles.intersection(target_roles):
                    continue

            matching_connections.append(connection)

        return matching_connections

    async def broadcast_message(self, message: RealtimeMessage) -> int:
        """Broadcast a message to all matching connections."""
        connections = self.get_connections_for_message(
            message.patient_id,
            message.risk_domain,
        )

        sent_count = 0
        for connection in connections:
            try:
                if connection.send_callback:
                    if asyncio.iscoroutinefunction(connection.send_callback):
                        if connection.client_type == "sse":
                            await connection.send_callback(message.to_sse())
                        else:
                            await connection.send_callback(message.to_json())
                    else:
                        if connection.client_type == "sse":
                            connection.send_callback(message.to_sse())
                        else:
                            connection.send_callback(message.to_json())
                    connection.message_count += 1
                    sent_count += 1
            except Exception as e:
                logger.error(f"Failed to send to {connection.connection_id}: {e}")
                # Consider unregistering failed connections
                self.unregister_connection(connection.connection_id)

        return sent_count

    async def queue_message(self, message: RealtimeMessage) -> None:
        """Queue a message for broadcast."""
        await self._message_queue.put(message)

    async def start_broadcast_loop(self) -> None:
        """Start the background broadcast loop."""
        self._running = True
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=30.0  # Send heartbeat every 30s
                )
                await self.broadcast_message(message)
            except asyncio.TimeoutError:
                # Send heartbeat to all connections
                await self._send_heartbeats()

    async def _send_heartbeats(self) -> None:
        """Send heartbeat to all connections."""
        heartbeat = RealtimeMessage(
            message_type=MessageType.HEARTBEAT,
            timestamp=datetime.now(timezone.utc),
            patient_id=None,
            risk_domain=None,
            data={"connections": len(self._connections)},
        )
        await self.broadcast_message(heartbeat)

    def stop_broadcast_loop(self) -> None:
        """Stop the broadcast loop."""
        self._running = False

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        websocket_count = len([c for c in self._connections.values() if c.client_type == "websocket"])
        sse_count = len([c for c in self._connections.values() if c.client_type == "sse"])

        return {
            "total_connections": len(self._connections),
            "websocket_connections": websocket_count,
            "sse_connections": sse_count,
            "patient_subscriptions": len(self._patient_subscriptions),
            "domain_subscriptions": len(self._domain_subscriptions),
            "role_subscriptions": len(self._role_subscriptions),
            "queue_size": self._message_queue.qsize(),
        }


# =============================================================================
# REAL-TIME HUB
# =============================================================================

class RealtimeHub:
    """
    Central hub for real-time notifications.

    Integrates with the alert pipeline to push updates to connected clients.
    """

    def __init__(self):
        self.connections = ConnectionManager()
        self._started = False

    async def start(self) -> None:
        """Start the real-time hub."""
        if self._started:
            return
        self._started = True
        asyncio.create_task(self.connections.start_broadcast_loop())
        logger.info("RealtimeHub started")

    def stop(self) -> None:
        """Stop the real-time hub."""
        self.connections.stop_broadcast_loop()
        self._started = False
        logger.info("RealtimeHub stopped")

    async def notify_alert_fired(
        self,
        patient_id: str,
        risk_domain: str,
        alert_data: Dict[str, Any],
    ) -> int:
        """Notify clients of a fired alert."""
        message = RealtimeMessage(
            message_type=MessageType.ALERT_FIRED,
            timestamp=datetime.now(timezone.utc),
            patient_id=patient_id,
            risk_domain=risk_domain,
            data=alert_data,
            priority=2 if alert_data.get("severity") == "CRITICAL" else 1,
        )
        return await self.connections.broadcast_message(message)

    async def notify_alert_suppressed(
        self,
        patient_id: str,
        risk_domain: str,
        suppression_data: Dict[str, Any],
    ) -> int:
        """Notify clients of a suppressed alert."""
        message = RealtimeMessage(
            message_type=MessageType.ALERT_SUPPRESSED,
            timestamp=datetime.now(timezone.utc),
            patient_id=patient_id,
            risk_domain=risk_domain,
            data=suppression_data,
            priority=0,
        )
        return await self.connections.broadcast_message(message)

    async def notify_state_change(
        self,
        patient_id: str,
        risk_domain: str,
        state_data: Dict[str, Any],
    ) -> int:
        """Notify clients of a state change."""
        message = RealtimeMessage(
            message_type=MessageType.STATE_CHANGE,
            timestamp=datetime.now(timezone.utc),
            patient_id=patient_id,
            risk_domain=risk_domain,
            data=state_data,
            priority=1,
        )
        return await self.connections.broadcast_message(message)

    async def notify_episode_opened(
        self,
        patient_id: str,
        risk_domain: str,
        episode_data: Dict[str, Any],
    ) -> int:
        """Notify clients of a new episode."""
        message = RealtimeMessage(
            message_type=MessageType.EPISODE_OPENED,
            timestamp=datetime.now(timezone.utc),
            patient_id=patient_id,
            risk_domain=risk_domain,
            data=episode_data,
            priority=1,
        )
        return await self.connections.broadcast_message(message)

    async def notify_episode_closed(
        self,
        patient_id: str,
        risk_domain: str,
        episode_data: Dict[str, Any],
    ) -> int:
        """Notify clients of a closed episode."""
        message = RealtimeMessage(
            message_type=MessageType.EPISODE_CLOSED,
            timestamp=datetime.now(timezone.utc),
            patient_id=patient_id,
            risk_domain=risk_domain,
            data=episode_data,
            priority=0,
        )
        return await self.connections.broadcast_message(message)

    async def notify_acknowledgment(
        self,
        patient_id: str,
        risk_domain: str,
        ack_data: Dict[str, Any],
    ) -> int:
        """Notify clients of an acknowledgment."""
        message = RealtimeMessage(
            message_type=MessageType.ACKNOWLEDGMENT,
            timestamp=datetime.now(timezone.utc),
            patient_id=patient_id,
            risk_domain=risk_domain,
            data=ack_data,
            priority=0,
        )
        return await self.connections.broadcast_message(message)

    async def notify_escalation(
        self,
        patient_id: str,
        risk_domain: str,
        escalation_data: Dict[str, Any],
    ) -> int:
        """Notify clients of an escalation."""
        message = RealtimeMessage(
            message_type=MessageType.ESCALATION,
            timestamp=datetime.now(timezone.utc),
            patient_id=patient_id,
            risk_domain=risk_domain,
            data=escalation_data,
            priority=2,
        )
        return await self.connections.broadcast_message(message)

    async def notify_system_status(self, status_data: Dict[str, Any]) -> int:
        """Broadcast system status to all clients."""
        message = RealtimeMessage(
            message_type=MessageType.SYSTEM_STATUS,
            timestamp=datetime.now(timezone.utc),
            patient_id=None,
            risk_domain=None,
            data=status_data,
            priority=0,
        )
        return await self.connections.broadcast_message(message)

    def get_dashboard_callback(self) -> Callable:
        """Get a callback function for pipeline integration."""
        async def callback(data: Dict[str, Any]) -> None:
            patient_id = data.get("patient_id")
            risk_domain = data.get("risk_domain")
            alert_fired = data.get("alert_fired", False)

            if alert_fired:
                await self.notify_alert_fired(patient_id, risk_domain, data)
            else:
                await self.notify_state_change(patient_id, risk_domain, data)

        return callback


# =============================================================================
# FASTAPI INTEGRATION HELPERS
# =============================================================================

async def websocket_handler(websocket: Any, hub: RealtimeHub) -> None:
    """
    Handle a WebSocket connection.

    Usage with FastAPI:
    ```
    @app.websocket("/ws/alerts")
    async def alerts_websocket(websocket: WebSocket):
        await websocket_handler(websocket, get_hub())
    ```
    """
    import uuid
    connection_id = f"ws_{uuid.uuid4().hex[:8]}"

    await websocket.accept()

    async def send_callback(data: str) -> None:
        await websocket.send_text(data)

    # Register connection
    connection = hub.connections.register_connection(
        connection_id=connection_id,
        client_type="websocket",
        send_callback=send_callback,
    )

    try:
        while True:
            # Handle incoming messages (subscription updates, etc.)
            data = await websocket.receive_text()
            try:
                message = json.loads(data)

                if message.get("type") == "subscribe":
                    hub.connections.update_subscription(
                        connection_id,
                        patient_ids=set(message.get("patient_ids", [])),
                        risk_domains=set(message.get("risk_domains", [])),
                        roles=set(message.get("roles", [])),
                    )
                    await websocket.send_text(json.dumps({
                        "type": "subscribed",
                        "filters": {
                            "patient_ids": list(connection.patient_ids),
                            "risk_domains": list(connection.risk_domains),
                            "roles": list(connection.roles),
                        }
                    }))
                elif message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

            except json.JSONDecodeError:
                pass

    except Exception as e:
        logger.debug(f"WebSocket {connection_id} disconnected: {e}")
    finally:
        hub.connections.unregister_connection(connection_id)


async def sse_generator(
    hub: RealtimeHub,
    patient_ids: Optional[List[str]] = None,
    risk_domains: Optional[List[str]] = None,
    roles: Optional[List[str]] = None,
):
    """
    Generate Server-Sent Events.

    Usage with FastAPI:
    ```
    @app.get("/sse/alerts")
    async def alerts_sse(
        patient_ids: List[str] = Query(None),
        risk_domains: List[str] = Query(None),
    ):
        return StreamingResponse(
            sse_generator(get_hub(), patient_ids, risk_domains),
            media_type="text/event-stream"
        )
    ```
    """
    import uuid
    connection_id = f"sse_{uuid.uuid4().hex[:8]}"
    message_queue: asyncio.Queue = asyncio.Queue()

    async def send_callback(data: str) -> None:
        await message_queue.put(data)

    # Register connection
    hub.connections.register_connection(
        connection_id=connection_id,
        client_type="sse",
        send_callback=send_callback,
        patient_ids=set(patient_ids) if patient_ids else set(),
        risk_domains=set(risk_domains) if risk_domains else set(),
        roles=set(roles) if roles else set(),
    )

    try:
        # Initial connection event
        yield f"event: connected\ndata: {{\"connection_id\": \"{connection_id}\"}}\n\n"

        while True:
            try:
                # Wait for message with timeout for heartbeat
                data = await asyncio.wait_for(message_queue.get(), timeout=30.0)
                yield data
            except asyncio.TimeoutError:
                # Send heartbeat
                yield f"event: heartbeat\ndata: {{\"time\": \"{datetime.now(timezone.utc).isoformat()}\"}}\n\n"

    except asyncio.CancelledError:
        pass
    finally:
        hub.connections.unregister_connection(connection_id)


# =============================================================================
# GLOBAL HUB INSTANCE
# =============================================================================

_hub: Optional[RealtimeHub] = None


def get_hub() -> RealtimeHub:
    """Get the global real-time hub."""
    global _hub
    if _hub is None:
        _hub = RealtimeHub()
    return _hub


def set_hub(hub: RealtimeHub) -> None:
    """Set a custom hub instance."""
    global _hub
    _hub = hub
