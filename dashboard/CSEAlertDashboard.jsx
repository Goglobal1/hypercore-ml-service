/**
 * CSE Alert Dashboard - Clinical State Engine Real-Time Dashboard
 *
 * Connects to HyperCore ML Service alert system for real-time clinical monitoring.
 *
 * Features:
 * - Real-time WebSocket/SSE connection for live updates
 * - 15 risk domain support
 * - Patient state visualization (S0-S3)
 * - Alert feed with filtering
 * - Statistics dashboard
 * - Biomarker trend charts
 *
 * @version 2.0.0
 * @requires React 18+
 * @requires recharts
 * @requires lucide-react
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Area, AreaChart
} from 'recharts';
import {
  Activity, AlertTriangle, Bell, BellOff, CheckCircle, Clock, Heart,
  RefreshCw, Settings, TrendingDown, TrendingUp, Users, Wifi, WifiOff,
  XCircle, Zap, Filter, ChevronDown, ChevronUp, Eye, EyeOff
} from 'lucide-react';

// =============================================================================
// CONFIGURATION
// =============================================================================

const API_BASE_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/alerts/ws';
const SSE_URL = 'http://localhost:8000/alerts/sse';

// All 15 risk domains from the merged alert system
const RISK_DOMAINS = [
  { id: 'sepsis', name: 'Sepsis', color: '#ef4444', icon: '🦠' },
  { id: 'cardiac', name: 'Cardiac', color: '#f97316', icon: '❤️' },
  { id: 'kidney', name: 'Kidney', color: '#eab308', icon: '🫘' },
  { id: 'respiratory', name: 'Respiratory', color: '#22c55e', icon: '🫁' },
  { id: 'hepatic', name: 'Hepatic', color: '#14b8a6', icon: '🔬' },
  { id: 'neurological', name: 'Neurological', color: '#3b82f6', icon: '🧠' },
  { id: 'metabolic', name: 'Metabolic', color: '#8b5cf6', icon: '⚗️' },
  { id: 'hematologic', name: 'Hematologic', color: '#ec4899', icon: '🩸' },
  { id: 'oncology', name: 'Oncology', color: '#6366f1', icon: '🎗️' },
  { id: 'multi_system', name: 'Multi-System', color: '#dc2626', icon: '⚠️' },
  { id: 'deterioration', name: 'Deterioration', color: '#b91c1c', icon: '📉' },
  { id: 'infection', name: 'Infection', color: '#ca8a04', icon: '🔴' },
  { id: 'outbreak', name: 'Outbreak', color: '#7c3aed', icon: '🌐' },
  { id: 'trial_confounder', name: 'Trial Confounder', color: '#0891b2', icon: '🧪' },
  { id: 'custom', name: 'Custom', color: '#64748b', icon: '⚙️' },
];

// Clinical state definitions
const CLINICAL_STATES = {
  S0: { name: 'Stable', color: '#22c55e', bgColor: '#dcfce7', description: 'Low risk, routine monitoring' },
  S1: { name: 'Watch', color: '#eab308', bgColor: '#fef9c3', description: 'Elevated risk, increased monitoring' },
  S2: { name: 'Escalating', color: '#f97316', bgColor: '#ffedd5', description: 'High risk, active intervention needed' },
  S3: { name: 'Critical', color: '#ef4444', bgColor: '#fee2e2', description: 'Critical, immediate action required' },
};

// Alert severity levels
const SEVERITY_LEVELS = {
  INFO: { color: '#3b82f6', bgColor: '#dbeafe', icon: 'info' },
  WARNING: { color: '#eab308', bgColor: '#fef9c3', icon: 'warning' },
  URGENT: { color: '#f97316', bgColor: '#ffedd5', icon: 'urgent' },
  CRITICAL: { color: '#ef4444', bgColor: '#fee2e2', icon: 'critical' },
};

// =============================================================================
// UTILITY HOOKS
// =============================================================================

/**
 * Custom hook for WebSocket connection with SSE fallback
 */
function useRealtimeConnection(onMessage) {
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [connectionType, setConnectionType] = useState(null);
  const wsRef = useRef(null);
  const sseRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  const connect = useCallback(() => {
    // Try WebSocket first
    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionStatus('connected');
        setConnectionType('websocket');
        console.log('[CSE Dashboard] WebSocket connected');

        // Subscribe to all domains
        ws.send(JSON.stringify({
          type: 'subscribe',
          risk_domains: RISK_DOMAINS.map(d => d.id),
          roles: ['dashboard']
        }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (e) {
          console.error('[CSE Dashboard] Failed to parse message:', e);
        }
      };

      ws.onerror = (error) => {
        console.warn('[CSE Dashboard] WebSocket error, falling back to SSE');
        ws.close();
        connectSSE();
      };

      ws.onclose = () => {
        if (connectionType === 'websocket') {
          setConnectionStatus('disconnected');
          scheduleReconnect();
        }
      };
    } catch (e) {
      console.warn('[CSE Dashboard] WebSocket not available, using SSE');
      connectSSE();
    }
  }, [onMessage, connectionType]);

  const connectSSE = useCallback(() => {
    try {
      const params = new URLSearchParams({
        risk_domains: RISK_DOMAINS.map(d => d.id).join(','),
        roles: 'dashboard'
      });
      const eventSource = new EventSource(`${SSE_URL}?${params}`);
      sseRef.current = eventSource;

      eventSource.onopen = () => {
        setConnectionStatus('connected');
        setConnectionType('sse');
        console.log('[CSE Dashboard] SSE connected');
      };

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (e) {
          console.error('[CSE Dashboard] Failed to parse SSE message:', e);
        }
      };

      eventSource.onerror = () => {
        setConnectionStatus('disconnected');
        eventSource.close();
        scheduleReconnect();
      };
    } catch (e) {
      console.error('[CSE Dashboard] SSE connection failed:', e);
      setConnectionStatus('error');
    }
  }, [onMessage]);

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    reconnectTimeoutRef.current = setTimeout(() => {
      console.log('[CSE Dashboard] Attempting reconnect...');
      connect();
    }, 5000);
  }, [connect]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (sseRef.current) {
      sseRef.current.close();
      sseRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    setConnectionStatus('disconnected');
    setConnectionType(null);
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, []);

  return { connectionStatus, connectionType, reconnect: connect, disconnect };
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

async function fetchHealth() {
  const response = await fetch(`${API_BASE_URL}/alerts/health`);
  return response.json();
}

async function fetchStats() {
  const response = await fetch(`${API_BASE_URL}/alerts/stats`);
  return response.json();
}

async function fetchPatientState(patientId) {
  const response = await fetch(`${API_BASE_URL}/alerts/patient/${patientId}/state`);
  if (response.status === 404) return null;
  return response.json();
}

async function fetchPatientDomainState(patientId, domain) {
  const response = await fetch(`${API_BASE_URL}/alerts/patient/${patientId}/state/${domain}`);
  if (response.status === 404) return null;
  return response.json();
}

async function fetchEvents(filters = {}) {
  const params = new URLSearchParams();
  if (filters.patient_id) params.append('patient_id', filters.patient_id);
  if (filters.risk_domain) params.append('risk_domain', filters.risk_domain);
  if (filters.event_type) params.append('event_type', filters.event_type);
  if (filters.limit) params.append('limit', filters.limit);

  const response = await fetch(`${API_BASE_URL}/alerts/events?${params}`);
  return response.json();
}

async function fetchDomainConfigs() {
  const response = await fetch(`${API_BASE_URL}/alerts/config/domains`);
  return response.json();
}

async function acknowledgeAlert(alertId, acknowledgedBy, actionTaken = null, notes = null) {
  const response = await fetch(`${API_BASE_URL}/alerts/acknowledge`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      alert_id: alertId,
      acknowledged_by: acknowledgedBy,
      action_taken: actionTaken,
      notes: notes,
      close_episode: false
    })
  });
  return response.json();
}

async function submitPatientIntake(data) {
  const response = await fetch(`${API_BASE_URL}/alerts/patient/intake`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return response.json();
}

// =============================================================================
// COMPONENTS
// =============================================================================

/**
 * Connection Status Indicator
 */
function ConnectionStatus({ status, type, onReconnect }) {
  const statusConfig = {
    connected: { icon: Wifi, color: 'text-green-500', bg: 'bg-green-100', text: 'Connected' },
    disconnected: { icon: WifiOff, color: 'text-red-500', bg: 'bg-red-100', text: 'Disconnected' },
    error: { icon: XCircle, color: 'text-red-500', bg: 'bg-red-100', text: 'Error' },
  };

  const config = statusConfig[status] || statusConfig.disconnected;
  const Icon = config.icon;

  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${config.bg}`}>
      <Icon className={`w-4 h-4 ${config.color}`} />
      <span className={`text-sm font-medium ${config.color}`}>
        {config.text} {type && `(${type.toUpperCase()})`}
      </span>
      {status !== 'connected' && (
        <button
          onClick={onReconnect}
          className="ml-2 p-1 hover:bg-white/50 rounded"
          title="Reconnect"
        >
          <RefreshCw className="w-3 h-3" />
        </button>
      )}
    </div>
  );
}

/**
 * Clinical State Badge
 */
function StateBadge({ state, size = 'md' }) {
  const config = CLINICAL_STATES[state] || CLINICAL_STATES.S0;
  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
    lg: 'px-4 py-2 text-base',
  };

  return (
    <span
      className={`inline-flex items-center font-semibold rounded-full ${sizeClasses[size]}`}
      style={{ backgroundColor: config.bgColor, color: config.color }}
    >
      {state} - {config.name}
    </span>
  );
}

/**
 * Severity Badge
 */
function SeverityBadge({ severity }) {
  const config = SEVERITY_LEVELS[severity] || SEVERITY_LEVELS.INFO;

  return (
    <span
      className="px-2 py-0.5 text-xs font-semibold rounded"
      style={{ backgroundColor: config.bgColor, color: config.color }}
    >
      {severity}
    </span>
  );
}

/**
 * Domain Badge
 */
function DomainBadge({ domain }) {
  const config = RISK_DOMAINS.find(d => d.id === domain) || { name: domain, color: '#64748b', icon: '?' };

  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded text-white"
      style={{ backgroundColor: config.color }}
    >
      <span>{config.icon}</span>
      {config.name}
    </span>
  );
}

/**
 * Trend Arrow
 */
function TrendArrow({ direction, value }) {
  if (direction === 'improving' || value < 0) {
    return <TrendingDown className="w-4 h-4 text-green-500" />;
  }
  if (direction === 'worsening' || value > 0) {
    return <TrendingUp className="w-4 h-4 text-red-500" />;
  }
  return <span className="w-4 h-4 text-gray-400">—</span>;
}

/**
 * Time-to-Harm Display
 */
function TimeToHarm({ hours, window }) {
  const getColor = () => {
    if (hours <= 1) return 'text-red-600 bg-red-100';
    if (hours <= 6) return 'text-orange-600 bg-orange-100';
    if (hours <= 24) return 'text-yellow-600 bg-yellow-100';
    return 'text-green-600 bg-green-100';
  };

  const formatTime = () => {
    if (hours < 1) return `${Math.round(hours * 60)} min`;
    if (hours < 24) return `${hours.toFixed(1)} hrs`;
    return `${Math.round(hours / 24)} days`;
  };

  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${getColor()}`}>
      <Clock className="w-4 h-4" />
      <span className="font-semibold">{formatTime()}</span>
      <span className="text-xs opacity-75">({window})</span>
    </div>
  );
}

/**
 * Patient State Panel
 */
function PatientStatePanel({ patientId, state, onRefresh }) {
  if (!state) {
    return (
      <div className="bg-gray-50 rounded-lg p-4 text-center text-gray-500">
        No state data for patient {patientId}
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-gray-900">Patient: {patientId}</h3>
        <button onClick={onRefresh} className="p-1 hover:bg-gray-100 rounded">
          <RefreshCw className="w-4 h-4 text-gray-500" />
        </button>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* State */}
        <div>
          <p className="text-xs text-gray-500 mb-1">Clinical State</p>
          <StateBadge state={state.state_now || 'S0'} size="lg" />
        </div>

        {/* Risk Score */}
        <div>
          <p className="text-xs text-gray-500 mb-1">Risk Score</p>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold">
              {((state.risk_score || 0) * 100).toFixed(0)}%
            </span>
            <TrendArrow value={state.velocity || 0} />
          </div>
        </div>

        {/* Time-to-Harm */}
        {state.time_to_harm && (
          <div className="col-span-2">
            <p className="text-xs text-gray-500 mb-1">Time to Harm</p>
            <TimeToHarm
              hours={state.time_to_harm.hours}
              window={state.time_to_harm.intervention_window}
            />
          </div>
        )}

        {/* Contributing Biomarkers */}
        {state.contributing_biomarkers && state.contributing_biomarkers.length > 0 && (
          <div className="col-span-2">
            <p className="text-xs text-gray-500 mb-1">Contributing Biomarkers</p>
            <div className="flex flex-wrap gap-1">
              {state.contributing_biomarkers.map((marker, i) => (
                <span key={i} className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded">
                  {marker}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Clinical Headline */}
        {state.clinical_headline && (
          <div className="col-span-2">
            <p className="text-xs text-gray-500 mb-1">Assessment</p>
            <p className="text-sm text-gray-700">{state.clinical_headline}</p>
          </div>
        )}

        {/* Suggested Action */}
        {state.suggested_action && (
          <div className="col-span-2 bg-blue-50 rounded p-2">
            <p className="text-xs text-blue-600 font-medium">Suggested Action</p>
            <p className="text-sm text-blue-800">{state.suggested_action}</p>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Alert Card
 */
function AlertCard({ alert, onAcknowledge, expanded, onToggleExpand }) {
  const [acknowledging, setAcknowledging] = useState(false);

  const handleAcknowledge = async () => {
    setAcknowledging(true);
    try {
      await onAcknowledge(alert.event_id);
    } finally {
      setAcknowledging(false);
    }
  };

  const severityConfig = SEVERITY_LEVELS[alert.severity] || SEVERITY_LEVELS.INFO;
  const isInterruptive = alert.alert_type === 'interruptive';

  return (
    <div
      className={`bg-white rounded-lg shadow-sm border-l-4 p-3 transition-all ${
        isInterruptive ? 'animate-pulse-slow' : ''
      }`}
      style={{ borderLeftColor: severityConfig.color }}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-semibold text-gray-900">{alert.patient_id}</span>
            <DomainBadge domain={alert.risk_domain} />
            <SeverityBadge severity={alert.severity} />
          </div>

          <p className="text-sm text-gray-600 truncate">
            {alert.clinical_headline || `State: ${alert.state_current}`}
          </p>

          <p className="text-xs text-gray-400 mt-1">
            {new Date(alert.timestamp).toLocaleString()}
          </p>
        </div>

        <div className="flex items-center gap-2">
          {isInterruptive && !alert.acknowledged && (
            <button
              onClick={handleAcknowledge}
              disabled={acknowledging}
              className="px-3 py-1 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 disabled:opacity-50"
            >
              {acknowledging ? 'ACK...' : 'ACK'}
            </button>
          )}
          <button
            onClick={onToggleExpand}
            className="p-1 hover:bg-gray-100 rounded"
          >
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="mt-3 pt-3 border-t text-sm space-y-2">
          <div className="grid grid-cols-2 gap-2">
            <div>
              <span className="text-gray-500">State Transition:</span>{' '}
              <span>{alert.state_previous || '—'} → {alert.state_current}</span>
            </div>
            <div>
              <span className="text-gray-500">Risk Score:</span>{' '}
              <span>{((alert.risk_score || 0) * 100).toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-gray-500">Confidence:</span>{' '}
              <span>{((alert.confidence || 0) * 100).toFixed(0)}%</span>
            </div>
            <div>
              <span className="text-gray-500">Episode:</span>{' '}
              <span>{alert.episode_id || '—'}</span>
            </div>
          </div>

          {alert.clinical_rationale && (
            <div className="bg-gray-50 rounded p-2">
              <p className="text-gray-600">{alert.clinical_rationale}</p>
            </div>
          )}

          {alert.recommendations && alert.recommendations.length > 0 && (
            <div>
              <p className="text-gray-500 font-medium mb-1">Recommendations:</p>
              <ul className="list-disc list-inside text-gray-600 space-y-0.5">
                {alert.recommendations.map((rec, i) => (
                  <li key={i}>{rec}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Alert Feed
 */
function AlertFeed({ alerts, onAcknowledge }) {
  const [expandedId, setExpandedId] = useState(null);
  const [filters, setFilters] = useState({
    severity: null,
    domain: null,
    acknowledged: null,
  });
  const [showFilters, setShowFilters] = useState(false);

  const filteredAlerts = alerts.filter(alert => {
    if (filters.severity && alert.severity !== filters.severity) return false;
    if (filters.domain && alert.risk_domain !== filters.domain) return false;
    if (filters.acknowledged === true && !alert.acknowledged) return false;
    if (filters.acknowledged === false && alert.acknowledged) return false;
    return true;
  });

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      <div className="p-3 border-b flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bell className="w-5 h-5 text-gray-500" />
          <h3 className="font-semibold">Alert Feed</h3>
          <span className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full">
            {filteredAlerts.length}
          </span>
        </div>
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={`p-1.5 rounded ${showFilters ? 'bg-blue-100 text-blue-600' : 'hover:bg-gray-100'}`}
        >
          <Filter className="w-4 h-4" />
        </button>
      </div>

      {showFilters && (
        <div className="p-3 bg-gray-50 border-b flex flex-wrap gap-2">
          <select
            value={filters.severity || ''}
            onChange={(e) => setFilters({ ...filters, severity: e.target.value || null })}
            className="px-2 py-1 text-sm border rounded"
          >
            <option value="">All Severities</option>
            {Object.keys(SEVERITY_LEVELS).map(sev => (
              <option key={sev} value={sev}>{sev}</option>
            ))}
          </select>

          <select
            value={filters.domain || ''}
            onChange={(e) => setFilters({ ...filters, domain: e.target.value || null })}
            className="px-2 py-1 text-sm border rounded"
          >
            <option value="">All Domains</option>
            {RISK_DOMAINS.map(d => (
              <option key={d.id} value={d.id}>{d.name}</option>
            ))}
          </select>

          <select
            value={filters.acknowledged === null ? '' : filters.acknowledged.toString()}
            onChange={(e) => setFilters({
              ...filters,
              acknowledged: e.target.value === '' ? null : e.target.value === 'true'
            })}
            className="px-2 py-1 text-sm border rounded"
          >
            <option value="">All Status</option>
            <option value="false">Unacknowledged</option>
            <option value="true">Acknowledged</option>
          </select>
        </div>
      )}

      <div className="max-h-[500px] overflow-y-auto p-3 space-y-2">
        {filteredAlerts.length === 0 ? (
          <p className="text-center text-gray-500 py-8">No alerts to display</p>
        ) : (
          filteredAlerts.map(alert => (
            <AlertCard
              key={alert.event_id}
              alert={alert}
              onAcknowledge={onAcknowledge}
              expanded={expandedId === alert.event_id}
              onToggleExpand={() => setExpandedId(
                expandedId === alert.event_id ? null : alert.event_id
              )}
            />
          ))
        )}
      </div>
    </div>
  );
}

/**
 * Statistics Panel
 */
function StatisticsPanel({ stats, isLoading }) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map(i => (
          <div key={i} className="bg-white rounded-lg shadow-sm border p-4 animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-1/2 mb-2"></div>
            <div className="h-8 bg-gray-200 rounded w-3/4"></div>
          </div>
        ))}
      </div>
    );
  }

  const storage = stats?.storage || {};
  const realtime = stats?.realtime || {};

  const statItems = [
    {
      label: 'Patients Tracked',
      value: storage.patient_states || 0,
      icon: Users,
      color: 'text-blue-500',
    },
    {
      label: 'Active Episodes',
      value: storage.open_episodes || 0,
      icon: Activity,
      color: 'text-orange-500',
    },
    {
      label: 'Events (24h)',
      value: storage.events || 0,
      icon: Bell,
      color: 'text-purple-500',
    },
    {
      label: 'Pending Escalations',
      value: stats?.escalations?.pending || 0,
      icon: AlertTriangle,
      color: 'text-red-500',
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {statItems.map((item, i) => {
        const Icon = item.icon;
        return (
          <div key={i} className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center gap-2 mb-2">
              <Icon className={`w-5 h-5 ${item.color}`} />
              <span className="text-sm text-gray-500">{item.label}</span>
            </div>
            <p className="text-2xl font-bold text-gray-900">{item.value}</p>
          </div>
        );
      })}
    </div>
  );
}

/**
 * Biomarker Trend Chart
 */
function BiomarkerTrendChart({ data, biomarker, thresholds }) {
  if (!data || data.length === 0) {
    return (
      <div className="h-40 flex items-center justify-center text-gray-400">
        No trend data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={160}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          dataKey="time"
          tick={{ fontSize: 10 }}
          tickFormatter={(t) => new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        />
        <YAxis tick={{ fontSize: 10 }} />
        <Tooltip />
        {thresholds?.warning && (
          <ReferenceLine y={thresholds.warning} stroke="#eab308" strokeDasharray="5 5" />
        )}
        {thresholds?.critical && (
          <ReferenceLine y={thresholds.critical} stroke="#ef4444" strokeDasharray="5 5" />
        )}
        <Area
          type="monotone"
          dataKey="value"
          stroke="#3b82f6"
          fill="#dbeafe"
          name={biomarker}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/**
 * Patient Search/Add Panel
 */
function PatientSearchPanel({ onSelectPatient, onSubmitIntake }) {
  const [patientId, setPatientId] = useState('');
  const [domain, setDomain] = useState('sepsis');
  const [showIntakeForm, setShowIntakeForm] = useState(false);
  const [labData, setLabData] = useState('');
  const [vitalsData, setVitalsData] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleSearch = () => {
    if (patientId.trim()) {
      onSelectPatient(patientId.trim());
    }
  };

  const handleSubmitIntake = async () => {
    setSubmitting(true);
    try {
      const data = {
        patient_id: patientId,
        risk_domain: domain,
        lab_data: labData ? JSON.parse(labData) : null,
        vitals_data: vitalsData ? JSON.parse(vitalsData) : null,
      };
      await onSubmitIntake(data);
      setShowIntakeForm(false);
    } catch (e) {
      alert('Error: ' + e.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border p-4">
      <div className="flex gap-2 mb-3">
        <input
          type="text"
          value={patientId}
          onChange={(e) => setPatientId(e.target.value)}
          placeholder="Enter Patient ID"
          className="flex-1 px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
        />
        <button
          onClick={handleSearch}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Search
        </button>
        <button
          onClick={() => setShowIntakeForm(!showIntakeForm)}
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
        >
          + Intake
        </button>
      </div>

      {showIntakeForm && (
        <div className="space-y-3 pt-3 border-t">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Risk Domain</label>
            <select
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
              className="w-full px-3 py-2 border rounded"
            >
              {RISK_DOMAINS.map(d => (
                <option key={d.id} value={d.id}>{d.icon} {d.name}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Lab Data (JSON)
            </label>
            <textarea
              value={labData}
              onChange={(e) => setLabData(e.target.value)}
              placeholder='{"lactate": 2.5, "WBC": 12.0}'
              className="w-full px-3 py-2 border rounded font-mono text-sm"
              rows={2}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Vitals Data (JSON)
            </label>
            <textarea
              value={vitalsData}
              onChange={(e) => setVitalsData(e.target.value)}
              placeholder='{"heart_rate": 90, "temperature": 37.5}'
              className="w-full px-3 py-2 border rounded font-mono text-sm"
              rows={2}
            />
          </div>

          <button
            onClick={handleSubmitIntake}
            disabled={submitting || !patientId}
            className="w-full py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50"
          >
            {submitting ? 'Processing...' : 'Submit Intake'}
          </button>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// MAIN DASHBOARD COMPONENT
// =============================================================================

export default function CSEAlertDashboard() {
  // State
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [patientState, setPatientState] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);

  // Handle real-time messages
  const handleRealtimeMessage = useCallback((message) => {
    console.log('[CSE Dashboard] Received:', message);
    setLastUpdate(new Date());

    if (message.type === 'alert' || message.type === 'evaluation') {
      setAlerts(prev => [message.data, ...prev].slice(0, 100));

      // Refresh patient state if it's the selected patient
      if (selectedPatient && message.data?.patient_id === selectedPatient) {
        loadPatientState(selectedPatient);
      }
    }

    if (message.type === 'acknowledgment') {
      setAlerts(prev => prev.map(a =>
        a.event_id === message.data?.alert_id
          ? { ...a, acknowledged: true }
          : a
      ));
    }
  }, [selectedPatient]);

  // Real-time connection
  const { connectionStatus, connectionType, reconnect } = useRealtimeConnection(handleRealtimeMessage);

  // Load initial data
  useEffect(() => {
    async function loadInitialData() {
      setIsLoading(true);
      try {
        const [healthData, statsData, eventsData] = await Promise.all([
          fetchHealth().catch(() => null),
          fetchStats().catch(() => null),
          fetchEvents({ limit: 50 }).catch(() => ({ events: [] })),
        ]);

        setHealth(healthData);
        setStats(statsData);
        setAlerts(eventsData.events || []);
      } catch (e) {
        console.error('[CSE Dashboard] Failed to load data:', e);
      } finally {
        setIsLoading(false);
      }
    }

    loadInitialData();

    // Refresh stats every 30 seconds
    const interval = setInterval(() => {
      fetchStats().then(setStats).catch(console.error);
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  // Load patient state
  const loadPatientState = async (patientId) => {
    try {
      const state = await fetchPatientState(patientId);
      setPatientState(state);
    } catch (e) {
      console.error('[CSE Dashboard] Failed to load patient state:', e);
      setPatientState(null);
    }
  };

  // Handle patient selection
  const handleSelectPatient = (patientId) => {
    setSelectedPatient(patientId);
    loadPatientState(patientId);
  };

  // Handle alert acknowledgment
  const handleAcknowledge = async (alertId) => {
    try {
      await acknowledgeAlert(alertId, 'dashboard_user');
      setAlerts(prev => prev.map(a =>
        a.event_id === alertId ? { ...a, acknowledged: true } : a
      ));
    } catch (e) {
      console.error('[CSE Dashboard] Acknowledge failed:', e);
    }
  };

  // Handle patient intake
  const handlePatientIntake = async (data) => {
    const result = await submitPatientIntake(data);
    if (result.success) {
      setSelectedPatient(data.patient_id);
      if (result.evaluation_result) {
        setPatientState(result.evaluation_result);
      }
      // Add to alerts if alert fired
      if (result.evaluation_result?.alert_event) {
        setAlerts(prev => [result.evaluation_result.alert_event, ...prev].slice(0, 100));
      }
    }
    return result;
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-8 h-8 text-blue-500" />
            <div>
              <h1 className="text-xl font-bold text-gray-900">CSE Alert Dashboard</h1>
              <p className="text-sm text-gray-500">Clinical State Engine Real-Time Monitoring</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {lastUpdate && (
              <span className="text-xs text-gray-400">
                Last update: {lastUpdate.toLocaleTimeString()}
              </span>
            )}
            <ConnectionStatus
              status={connectionStatus}
              type={connectionType}
              onReconnect={reconnect}
            />
            {health && (
              <div className={`flex items-center gap-1 px-2 py-1 rounded ${
                health.status === 'healthy' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
              }`}>
                {health.status === 'healthy' ? (
                  <CheckCircle className="w-4 h-4" />
                ) : (
                  <XCircle className="w-4 h-4" />
                )}
                <span className="text-sm font-medium">
                  {health.status === 'healthy' ? 'System Healthy' : 'System Error'}
                </span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
        {/* Statistics */}
        <StatisticsPanel stats={stats} isLoading={isLoading} />

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Patient */}
          <div className="space-y-4">
            <PatientSearchPanel
              onSelectPatient={handleSelectPatient}
              onSubmitIntake={handlePatientIntake}
            />

            {selectedPatient && (
              <PatientStatePanel
                patientId={selectedPatient}
                state={patientState}
                onRefresh={() => loadPatientState(selectedPatient)}
              />
            )}

            {/* Domain Quick Access */}
            <div className="bg-white rounded-lg shadow-sm border p-4">
              <h3 className="font-semibold mb-3">Risk Domains</h3>
              <div className="grid grid-cols-3 gap-2">
                {RISK_DOMAINS.map(domain => (
                  <div
                    key={domain.id}
                    className="flex flex-col items-center p-2 rounded hover:bg-gray-50 cursor-pointer"
                    title={domain.name}
                  >
                    <span className="text-xl">{domain.icon}</span>
                    <span className="text-xs text-gray-600 truncate w-full text-center">
                      {domain.name}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column - Alerts */}
          <div className="lg:col-span-2">
            <AlertFeed
              alerts={alerts}
              onAcknowledge={handleAcknowledge}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-8 py-4">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-500">
          HyperCore ML Service - CSE Alert Dashboard v2.0.0
        </div>
      </footer>
    </div>
  );
}
