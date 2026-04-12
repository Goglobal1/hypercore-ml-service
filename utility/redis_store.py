"""
Redis persistence layer for Utility Engine
"""

import os
import json
import redis
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from urllib.parse import urlparse


def get_redis_url() -> str:
    """Get Redis URL from environment, read at runtime not import time.
    Railway uses REDIS_PRIVATE_URL for internal connections.
    """
    return (
        os.environ.get('REDIS_PRIVATE_URL') or
        os.environ.get('REDIS_URL') or
        'redis://localhost:6379'
    )


class RedisStore:
    """
    Redis-backed storage for clinical events, feedback, and suppressed alerts.
    Falls back to in-memory if Redis unavailable.
    """

    def __init__(self):
        self._redis = None
        self._fallback = {}  # In-memory fallback
        self._redis_url = None
        self._connect()

    def _connect(self):
        # Read REDIS_URL at connection time, not import time
        self._redis_url = get_redis_url()

        # Log connection attempt (mask password for security)
        try:
            parsed = urlparse(self._redis_url)
            masked_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 6379}"
            is_railway = 'railway' in (parsed.hostname or '')
            print(f"[HYPERCORE] Redis connecting to: {masked_url} (Railway: {is_railway})")
        except:
            print(f"[HYPERCORE] Redis URL configured: {bool(self._redis_url)}")

        try:
            self._redis = redis.from_url(self._redis_url, decode_responses=True)
            self._redis.ping()
            print("[HYPERCORE] Redis connected for Utility Engine persistence")
        except Exception as e:
            print(f"[HYPERCORE] Redis unavailable, using in-memory storage: {e}")
            self._redis = None

    def _serialize(self, obj: Any) -> str:
        """Serialize object to JSON string."""
        if hasattr(obj, 'to_dict'):
            return json.dumps(obj.to_dict())
        return json.dumps(obj)

    def _deserialize(self, data: str) -> Dict:
        """Deserialize JSON string to dict."""
        return json.loads(data) if data else None

    # ==================== CLINICAL EVENTS ====================

    def save_event(self, event: Dict) -> bool:
        """Save clinical event."""
        key = f"event:{event['id']}"
        patient_key = f"patient_events:{event['patient_id']}"

        try:
            if self._redis:
                self._redis.set(key, self._serialize(event), ex=86400 * 7)  # 7 day expiry
                self._redis.sadd(patient_key, event['id'])
                self._redis.expire(patient_key, 86400 * 7)
                return True
            else:
                self._fallback[key] = event
                if patient_key not in self._fallback:
                    self._fallback[patient_key] = set()
                self._fallback[patient_key].add(event['id'])
                return True
        except Exception as e:
            print(f"[HYPERCORE] Error saving event: {e}")
            return False

    def get_event(self, event_id: str) -> Optional[Dict]:
        """Get event by ID."""
        key = f"event:{event_id}"

        try:
            if self._redis:
                data = self._redis.get(key)
                return self._deserialize(data)
            else:
                return self._fallback.get(key)
        except Exception as e:
            print(f"[HYPERCORE] Error getting event: {e}")
            return None

    def get_patient_events(self, patient_id: str, status: Optional[str] = None) -> List[Dict]:
        """Get all events for a patient."""
        patient_key = f"patient_events:{patient_id}"
        events = []

        try:
            if self._redis:
                event_ids = self._redis.smembers(patient_key)
                for event_id in event_ids:
                    event = self.get_event(event_id)
                    if event:
                        if status is None or event.get('status') == status:
                            events.append(event)
            else:
                event_ids = self._fallback.get(patient_key, set())
                for event_id in event_ids:
                    event = self._fallback.get(f"event:{event_id}")
                    if event:
                        if status is None or event.get('status') == status:
                            events.append(event)
        except Exception as e:
            print(f"[HYPERCORE] Error getting patient events: {e}")

        return events

    def find_matching_event(self, patient_id: str, event_type: str, endpoint: str) -> Optional[Dict]:
        """Find active event matching type or endpoint."""
        events = self.get_patient_events(patient_id, status='active')

        for event in events:
            if event.get('event_type') == event_type or event.get('primary_endpoint') == endpoint:
                return event

        return None

    # ==================== FEEDBACK ====================

    def save_feedback(self, feedback: Dict) -> bool:
        """Save alert feedback."""
        key = f"feedback:{feedback['id']}"
        alert_key = f"alert_feedback:{feedback['alert_id']}"
        patient_key = f"patient_feedback:{feedback['patient_id']}"

        try:
            if self._redis:
                self._redis.set(key, self._serialize(feedback), ex=86400 * 30)  # 30 day expiry
                self._redis.set(alert_key, feedback['id'], ex=86400 * 30)
                self._redis.lpush(patient_key, feedback['id'])
                self._redis.ltrim(patient_key, 0, 999)  # Keep last 1000
                self._redis.expire(patient_key, 86400 * 30)

                # Track for metrics
                self._increment_metric('total_alerts')
                if feedback.get('emission_decision') == 'fire':
                    self._increment_metric('fired_alerts')
                else:
                    self._increment_metric('suppressed_alerts')

                return True
            else:
                self._fallback[key] = feedback
                self._fallback[alert_key] = feedback['id']
                if patient_key not in self._fallback:
                    self._fallback[patient_key] = []
                self._fallback[patient_key].insert(0, feedback['id'])
                return True
        except Exception as e:
            print(f"[HYPERCORE] Error saving feedback: {e}")
            return False

    def get_feedback(self, feedback_id: str) -> Optional[Dict]:
        """Get feedback by ID."""
        key = f"feedback:{feedback_id}"

        try:
            if self._redis:
                data = self._redis.get(key)
                return self._deserialize(data)
            else:
                return self._fallback.get(key)
        except Exception as e:
            print(f"[HYPERCORE] Error getting feedback: {e}")
            return None

    def get_feedback_by_alert(self, alert_id: str) -> Optional[Dict]:
        """Get feedback for an alert."""
        alert_key = f"alert_feedback:{alert_id}"

        try:
            if self._redis:
                feedback_id = self._redis.get(alert_key)
                if feedback_id:
                    return self.get_feedback(feedback_id)
            else:
                feedback_id = self._fallback.get(alert_key)
                if feedback_id:
                    return self._fallback.get(f"feedback:{feedback_id}")
        except Exception as e:
            print(f"[HYPERCORE] Error getting feedback by alert: {e}")

        return None

    def update_feedback(self, feedback_id: str, updates: Dict) -> bool:
        """Update feedback record."""
        feedback = self.get_feedback(feedback_id)
        if feedback:
            feedback.update(updates)
            return self.save_feedback(feedback)
        return False

    def get_patient_feedback(self, patient_id: str, hours: int = 24) -> List[Dict]:
        """Get recent feedback for patient."""
        patient_key = f"patient_feedback:{patient_id}"
        feedback_list = []
        cutoff = datetime.now() - timedelta(hours=hours)

        try:
            if self._redis:
                feedback_ids = self._redis.lrange(patient_key, 0, 100)
                for fid in feedback_ids:
                    fb = self.get_feedback(fid)
                    if fb:
                        fired_at = datetime.fromisoformat(fb.get('fired_at', datetime.now().isoformat()))
                        if fired_at > cutoff:
                            feedback_list.append(fb)
            else:
                feedback_ids = self._fallback.get(patient_key, [])
                for fid in feedback_ids[:100]:
                    fb = self._fallback.get(f"feedback:{fid}")
                    if fb:
                        fired_at = datetime.fromisoformat(fb.get('fired_at', datetime.now().isoformat()))
                        if fired_at > cutoff:
                            feedback_list.append(fb)
        except Exception as e:
            print(f"[HYPERCORE] Error getting patient feedback: {e}")

        return feedback_list

    # ==================== SUPPRESSED ALERTS ====================

    def save_suppressed(self, suppressed: Dict) -> bool:
        """Save suppressed alert record."""
        key = f"suppressed:{suppressed['id']}"
        user_key = f"user_suppressed:{suppressed['user_id']}"

        try:
            if self._redis:
                self._redis.set(key, self._serialize(suppressed), ex=86400 * 7)
                self._redis.lpush(user_key, suppressed['id'])
                self._redis.ltrim(user_key, 0, 499)
                self._redis.expire(user_key, 86400 * 7)
                return True
            else:
                self._fallback[key] = suppressed
                if user_key not in self._fallback:
                    self._fallback[user_key] = []
                self._fallback[user_key].insert(0, suppressed['id'])
                return True
        except Exception as e:
            print(f"[HYPERCORE] Error saving suppressed: {e}")
            return False

    def get_user_suppressed(self, user_id: str, hours: int = 24) -> List[Dict]:
        """Get suppressed alerts for user."""
        user_key = f"user_suppressed:{user_id}"
        suppressed_list = []
        cutoff = datetime.now() - timedelta(hours=hours)

        try:
            if self._redis:
                suppressed_ids = self._redis.lrange(user_key, 0, 100)
                for sid in suppressed_ids:
                    s = self.get_suppressed(sid)
                    if s:
                        suppressed_at = datetime.fromisoformat(s.get('suppressed_at', datetime.now().isoformat()))
                        if suppressed_at > cutoff:
                            suppressed_list.append(s)
            else:
                suppressed_ids = self._fallback.get(user_key, [])
                for sid in suppressed_ids[:100]:
                    s = self._fallback.get(f"suppressed:{sid}")
                    if s:
                        suppressed_at = datetime.fromisoformat(s.get('suppressed_at', datetime.now().isoformat()))
                        if suppressed_at > cutoff:
                            suppressed_list.append(s)
        except Exception as e:
            print(f"[HYPERCORE] Error getting user suppressed: {e}")

        return suppressed_list

    def get_suppressed(self, suppressed_id: str) -> Optional[Dict]:
        """Get suppressed alert by ID."""
        key = f"suppressed:{suppressed_id}"

        try:
            if self._redis:
                data = self._redis.get(key)
                return self._deserialize(data)
            else:
                return self._fallback.get(key)
        except Exception as e:
            return None

    # ==================== METRICS ====================

    def _increment_metric(self, metric: str, amount: int = 1):
        """Increment a metric counter."""
        date_key = datetime.now().strftime('%Y-%m-%d')
        key = f"metric:{metric}:{date_key}"

        try:
            if self._redis:
                self._redis.incr(key, amount)
                self._redis.expire(key, 86400 * 30)
        except Exception as e:
            pass

    def get_metrics(self, hours: int = 168) -> Dict:
        """Get aggregated metrics."""
        metrics = {
            'total_alerts': 0,
            'fired': 0,
            'suppressed': 0,
            'acknowledged': 0,
            'action_taken': 0
        }

        try:
            if self._redis:
                # Get metrics for each day in range
                days = hours // 24 + 1
                for i in range(days):
                    date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')

                    total = self._redis.get(f"metric:total_alerts:{date}")
                    fired = self._redis.get(f"metric:fired_alerts:{date}")
                    suppressed = self._redis.get(f"metric:suppressed_alerts:{date}")
                    acked = self._redis.get(f"metric:acknowledged:{date}")
                    acted = self._redis.get(f"metric:action_taken:{date}")

                    metrics['total_alerts'] += int(total or 0)
                    metrics['fired'] += int(fired or 0)
                    metrics['suppressed'] += int(suppressed or 0)
                    metrics['acknowledged'] += int(acked or 0)
                    metrics['action_taken'] += int(acted or 0)
        except Exception as e:
            print(f"[HYPERCORE] Error getting metrics: {e}")

        # Calculate rates
        if metrics['fired'] > 0:
            metrics['acknowledgment_rate'] = metrics['acknowledged'] / metrics['fired']
            metrics['actionable_rate'] = metrics['action_taken'] / metrics['fired']
        else:
            metrics['acknowledgment_rate'] = 0
            metrics['actionable_rate'] = 0

        if metrics['total_alerts'] > 0:
            metrics['suppression_rate'] = metrics['suppressed'] / metrics['total_alerts']
        else:
            metrics['suppression_rate'] = 0

        metrics['redundant_rate'] = 0  # Need to track this separately
        metrics['ignored_rate'] = 1 - metrics['acknowledgment_rate'] if metrics['fired'] > 0 else 0

        return metrics

    def record_acknowledgment(self, feedback_id: str):
        """Record that an alert was acknowledged."""
        self._increment_metric('acknowledged')

    def record_action(self, feedback_id: str):
        """Record that action was taken."""
        self._increment_metric('action_taken')


# Singleton instance
_store = None

def get_redis_store() -> RedisStore:
    global _store
    if _store is None:
        _store = RedisStore()
    # Try to reconnect if not connected but URL is available
    elif _store._redis is None:
        current_url = get_redis_url()
        if current_url != 'redis://localhost:6379':
            print(f"[HYPERCORE] Redis URL available, attempting reconnect...")
            _store._connect()
    return _store


def reset_redis_store() -> RedisStore:
    """Force reset and reconnect the Redis store singleton."""
    global _store
    _store = RedisStore()
    return _store
