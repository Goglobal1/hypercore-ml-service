/**
 * CSE Alert Dashboard - Production Version
 *
 * This version uses environment variables for configuration.
 * Set the following in your .env file:
 *
 * REACT_APP_HYPERCORE_API_URL=https://your-api.com
 * REACT_APP_HYPERCORE_WS_URL=wss://your-api.com/alerts/ws
 * REACT_APP_HYPERCORE_SSE_URL=https://your-api.com/alerts/sse
 *
 * @version 2.0.0
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
// CONFIGURATION - Uses Environment Variables
// =============================================================================

const CONFIG = {
  API_BASE_URL: process.env.REACT_APP_HYPERCORE_API_URL || 'http://localhost:8000',
  WS_URL: process.env.REACT_APP_HYPERCORE_WS_URL || 'ws://localhost:8000/alerts/ws',
  SSE_URL: process.env.REACT_APP_HYPERCORE_SSE_URL || 'http://localhost:8000/alerts/sse',
  RECONNECT_INTERVAL: parseInt(process.env.REACT_APP_RECONNECT_INTERVAL || '5000'),
  STATS_REFRESH_INTERVAL: parseInt(process.env.REACT_APP_STATS_REFRESH_INTERVAL || '30000'),
  MAX_ALERTS_DISPLAY: parseInt(process.env.REACT_APP_MAX_ALERTS_DISPLAY || '100'),
};

// Export config for external use
export { CONFIG };

// All 15 risk domains from the merged alert system
export const RISK_DOMAINS = [
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
export const CLINICAL_STATES = {
  S0: { name: 'Stable', color: '#22c55e', bgColor: '#dcfce7', description: 'Low risk, routine monitoring' },
  S1: { name: 'Watch', color: '#eab308', bgColor: '#fef9c3', description: 'Elevated risk, increased monitoring' },
  S2: { name: 'Escalating', color: '#f97316', bgColor: '#ffedd5', description: 'High risk, active intervention needed' },
  S3: { name: 'Critical', color: '#ef4444', bgColor: '#fee2e2', description: 'Critical, immediate action required' },
};

// Alert severity levels
export const SEVERITY_LEVELS = {
  INFO: { color: '#3b82f6', bgColor: '#dbeafe', icon: 'info' },
  WARNING: { color: '#eab308', bgColor: '#fef9c3', icon: 'warning' },
  URGENT: { color: '#f97316', bgColor: '#ffedd5', icon: 'urgent' },
  CRITICAL: { color: '#ef4444', bgColor: '#fee2e2', icon: 'critical' },
};

// =============================================================================
// API SERVICE
// =============================================================================

export const AlertAPI = {
  async fetchHealth() {
    const response = await fetch(`${CONFIG.API_BASE_URL}/alerts/health`);
    if (!response.ok) throw new Error(`Health check failed: ${response.status}`);
    return response.json();
  },

  async fetchStats() {
    const response = await fetch(`${CONFIG.API_BASE_URL}/alerts/stats`);
    if (!response.ok) throw new Error(`Stats fetch failed: ${response.status}`);
    return response.json();
  },

  async fetchPatientState(patientId) {
    const response = await fetch(`${CONFIG.API_BASE_URL}/alerts/patient/${patientId}/state`);
    if (response.status === 404) return null;
    if (!response.ok) throw new Error(`Patient state fetch failed: ${response.status}`);
    return response.json();
  },

  async fetchPatientDomainState(patientId, domain) {
    const response = await fetch(`${CONFIG.API_BASE_URL}/alerts/patient/${patientId}/state/${domain}`);
    if (response.status === 404) return null;
    if (!response.ok) throw new Error(`Patient domain state fetch failed: ${response.status}`);
    return response.json();
  },

  async fetchEvents(filters = {}) {
    const params = new URLSearchParams();
    if (filters.patient_id) params.append('patient_id', filters.patient_id);
    if (filters.risk_domain) params.append('risk_domain', filters.risk_domain);
    if (filters.event_type) params.append('event_type', filters.event_type);
    if (filters.limit) params.append('limit', filters.limit);

    const response = await fetch(`${CONFIG.API_BASE_URL}/alerts/events?${params}`);
    if (!response.ok) throw new Error(`Events fetch failed: ${response.status}`);
    return response.json();
  },

  async fetchDomainConfigs() {
    const response = await fetch(`${CONFIG.API_BASE_URL}/alerts/config/domains`);
    if (!response.ok) throw new Error(`Domain configs fetch failed: ${response.status}`);
    return response.json();
  },

  async acknowledgeAlert(alertId, acknowledgedBy, options = {}) {
    const response = await fetch(`${CONFIG.API_BASE_URL}/alerts/acknowledge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        alert_id: alertId,
        acknowledged_by: acknowledgedBy,
        action_taken: options.actionTaken || null,
        notes: options.notes || null,
        close_episode: options.closeEpisode || false
      })
    });
    if (!response.ok) throw new Error(`Acknowledge failed: ${response.status}`);
    return response.json();
  },

  async submitPatientIntake(data) {
    const response = await fetch(`${CONFIG.API_BASE_URL}/alerts/patient/intake`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!response.ok) throw new Error(`Patient intake failed: ${response.status}`);
    return response.json();
  },
};

// =============================================================================
// REAL-TIME CONNECTION HOOK
// =============================================================================

export function useRealtimeConnection(onMessage, options = {}) {
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [connectionType, setConnectionType] = useState(null);
  const wsRef = useRef(null);
  const sseRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  const {
    autoConnect = true,
    reconnectInterval = CONFIG.RECONNECT_INTERVAL,
    subscribeOnConnect = true,
  } = options;

  const connect = useCallback(() => {
    // Try WebSocket first
    try {
      const ws = new WebSocket(CONFIG.WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionStatus('connected');
        setConnectionType('websocket');
        console.log('[CSE Dashboard] WebSocket connected');

        if (subscribeOnConnect) {
          ws.send(JSON.stringify({
            type: 'subscribe',
            risk_domains: RISK_DOMAINS.map(d => d.id),
            roles: ['dashboard']
          }));
        }
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
  }, [onMessage, connectionType, subscribeOnConnect]);

  const connectSSE = useCallback(() => {
    try {
      const params = new URLSearchParams({
        risk_domains: RISK_DOMAINS.map(d => d.id).join(','),
        roles: 'dashboard'
      });
      const eventSource = new EventSource(`${CONFIG.SSE_URL}?${params}`);
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
    }, reconnectInterval);
  }, [connect, reconnectInterval]);

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

  const sendMessage = useCallback((message) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, []);

  useEffect(() => {
    if (autoConnect) {
      connect();
    }
    return () => disconnect();
  }, [autoConnect]);

  return {
    connectionStatus,
    connectionType,
    connect,
    disconnect,
    reconnect: connect,
    sendMessage,
  };
}

// =============================================================================
// COMPONENT EXPORTS
// =============================================================================

// Export individual components for flexible usage
export { default as ConnectionStatus } from './components/ConnectionStatus';
export { default as StateBadge } from './components/StateBadge';
export { default as SeverityBadge } from './components/SeverityBadge';
export { default as DomainBadge } from './components/DomainBadge';
export { default as TimeToHarm } from './components/TimeToHarm';
export { default as PatientStatePanel } from './components/PatientStatePanel';
export { default as AlertCard } from './components/AlertCard';
export { default as AlertFeed } from './components/AlertFeed';
export { default as StatisticsPanel } from './components/StatisticsPanel';
export { default as PatientSearchPanel } from './components/PatientSearchPanel';

// =============================================================================
// MAIN DASHBOARD COMPONENT
// =============================================================================

/**
 * CSEAlertDashboard - Main Dashboard Component
 *
 * @param {Object} props
 * @param {string} props.apiBaseUrl - Override API base URL
 * @param {string} props.wsUrl - Override WebSocket URL
 * @param {Function} props.onAlert - Callback when alert received
 * @param {Function} props.onStateChange - Callback when patient state changes
 * @param {string} props.currentUser - Current user ID for acknowledgments
 * @param {boolean} props.showHeader - Show/hide header (default: true)
 * @param {boolean} props.showFooter - Show/hide footer (default: true)
 */
export default function CSEAlertDashboard({
  apiBaseUrl,
  wsUrl,
  onAlert,
  onStateChange,
  currentUser = 'dashboard_user',
  showHeader = true,
  showFooter = true,
}) {
  // Override config if props provided
  if (apiBaseUrl) CONFIG.API_BASE_URL = apiBaseUrl;
  if (wsUrl) CONFIG.WS_URL = wsUrl;

  // State
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [patientState, setPatientState] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [error, setError] = useState(null);

  // Handle real-time messages
  const handleRealtimeMessage = useCallback((message) => {
    console.log('[CSE Dashboard] Received:', message);
    setLastUpdate(new Date());
    setError(null);

    if (message.type === 'alert' || message.type === 'evaluation') {
      const alertData = message.data;
      setAlerts(prev => [alertData, ...prev].slice(0, CONFIG.MAX_ALERTS_DISPLAY));

      // External callback
      if (onAlert) onAlert(alertData);

      // Refresh patient state if selected
      if (selectedPatient && alertData?.patient_id === selectedPatient) {
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
  }, [selectedPatient, onAlert]);

  // Real-time connection
  const { connectionStatus, connectionType, reconnect } = useRealtimeConnection(handleRealtimeMessage);

  // Load initial data
  useEffect(() => {
    async function loadInitialData() {
      setIsLoading(true);
      setError(null);
      try {
        const [healthData, statsData, eventsData] = await Promise.all([
          AlertAPI.fetchHealth().catch(() => null),
          AlertAPI.fetchStats().catch(() => null),
          AlertAPI.fetchEvents({ limit: 50 }).catch(() => ({ events: [] })),
        ]);

        setHealth(healthData);
        setStats(statsData);
        setAlerts(eventsData.events || []);
      } catch (e) {
        console.error('[CSE Dashboard] Failed to load data:', e);
        setError(e.message);
      } finally {
        setIsLoading(false);
      }
    }

    loadInitialData();

    // Refresh stats periodically
    const interval = setInterval(() => {
      AlertAPI.fetchStats().then(setStats).catch(console.error);
    }, CONFIG.STATS_REFRESH_INTERVAL);

    return () => clearInterval(interval);
  }, []);

  // Load patient state
  const loadPatientState = async (patientId) => {
    try {
      const state = await AlertAPI.fetchPatientState(patientId);
      setPatientState(state);
      if (onStateChange) onStateChange(patientId, state);
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
      await AlertAPI.acknowledgeAlert(alertId, currentUser);
      setAlerts(prev => prev.map(a =>
        a.event_id === alertId ? { ...a, acknowledged: true } : a
      ));
    } catch (e) {
      console.error('[CSE Dashboard] Acknowledge failed:', e);
      throw e;
    }
  };

  // Handle patient intake
  const handlePatientIntake = async (data) => {
    const result = await AlertAPI.submitPatientIntake(data);
    if (result.success) {
      setSelectedPatient(data.patient_id);
      if (result.evaluation_result) {
        setPatientState(result.evaluation_result);
      }
      if (result.evaluation_result?.alert_event) {
        setAlerts(prev => [result.evaluation_result.alert_event, ...prev].slice(0, CONFIG.MAX_ALERTS_DISPLAY));
      }
    }
    return result;
  };

  // Render loading state
  if (isLoading && !health) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin text-blue-500 mx-auto mb-4" />
          <p className="text-gray-500">Loading CSE Dashboard...</p>
        </div>
      </div>
    );
  }

  // Render error state
  if (error && !health) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center bg-white p-8 rounded-lg shadow-lg">
          <XCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Connection Error</h2>
          <p className="text-gray-500 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      {showHeader && (
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
              <ConnectionStatusComponent
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
      )}

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
        {/* Statistics */}
        <StatisticsPanelComponent stats={stats} isLoading={isLoading} />

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Patient */}
          <div className="space-y-4">
            <PatientSearchPanelComponent
              onSelectPatient={handleSelectPatient}
              onSubmitIntake={handlePatientIntake}
            />

            {selectedPatient && (
              <PatientStatePanelComponent
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
            <AlertFeedComponent
              alerts={alerts}
              onAcknowledge={handleAcknowledge}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      {showFooter && (
        <footer className="bg-white border-t mt-8 py-4">
          <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-500">
            HyperCore ML Service - CSE Alert Dashboard v2.0.0
          </div>
        </footer>
      )}
    </div>
  );
}

// Inline component definitions for production build
// (In a real project, these would be in separate files)

function ConnectionStatusComponent({ status, type, onReconnect }) {
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
        <button onClick={onReconnect} className="ml-2 p-1 hover:bg-white/50 rounded">
          <RefreshCw className="w-3 h-3" />
        </button>
      )}
    </div>
  );
}

function StatisticsPanelComponent({ stats, isLoading }) {
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
  const statItems = [
    { label: 'Patients Tracked', value: storage.patient_states || 0, icon: Users, color: 'text-blue-500' },
    { label: 'Active Episodes', value: storage.open_episodes || 0, icon: Activity, color: 'text-orange-500' },
    { label: 'Events (24h)', value: storage.events || 0, icon: Bell, color: 'text-purple-500' },
    { label: 'Pending Escalations', value: stats?.escalations?.pending || 0, icon: AlertTriangle, color: 'text-red-500' },
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

function PatientSearchPanelComponent({ onSelectPatient, onSubmitIntake }) {
  const [patientId, setPatientId] = useState('');
  const [domain, setDomain] = useState('sepsis');
  const [showIntakeForm, setShowIntakeForm] = useState(false);
  const [labData, setLabData] = useState('');
  const [vitalsData, setVitalsData] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleSearch = () => {
    if (patientId.trim()) onSelectPatient(patientId.trim());
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
        <button onClick={handleSearch} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
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
            <select value={domain} onChange={(e) => setDomain(e.target.value)} className="w-full px-3 py-2 border rounded">
              {RISK_DOMAINS.map(d => (<option key={d.id} value={d.id}>{d.icon} {d.name}</option>))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Lab Data (JSON)</label>
            <textarea value={labData} onChange={(e) => setLabData(e.target.value)} placeholder='{"lactate": 2.5}' className="w-full px-3 py-2 border rounded font-mono text-sm" rows={2} />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Vitals Data (JSON)</label>
            <textarea value={vitalsData} onChange={(e) => setVitalsData(e.target.value)} placeholder='{"heart_rate": 90}' className="w-full px-3 py-2 border rounded font-mono text-sm" rows={2} />
          </div>
          <button onClick={handleSubmitIntake} disabled={submitting || !patientId} className="w-full py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50">
            {submitting ? 'Processing...' : 'Submit Intake'}
          </button>
        </div>
      )}
    </div>
  );
}

function PatientStatePanelComponent({ patientId, state, onRefresh }) {
  if (!state) {
    return (<div className="bg-gray-50 rounded-lg p-4 text-center text-gray-500">No state data for patient {patientId}</div>);
  }
  const stateConfig = CLINICAL_STATES[state.state_now] || CLINICAL_STATES.S0;

  return (
    <div className="bg-white rounded-lg shadow-sm border p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-gray-900">Patient: {patientId}</h3>
        <button onClick={onRefresh} className="p-1 hover:bg-gray-100 rounded"><RefreshCw className="w-4 h-4 text-gray-500" /></button>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-gray-500 mb-1">Clinical State</p>
          <span className="inline-flex items-center font-semibold rounded-full px-4 py-2" style={{ backgroundColor: stateConfig.bgColor, color: stateConfig.color }}>
            {state.state_now} - {stateConfig.name}
          </span>
        </div>
        <div>
          <p className="text-xs text-gray-500 mb-1">Risk Score</p>
          <span className="text-2xl font-bold">{((state.risk_score || 0) * 100).toFixed(0)}%</span>
        </div>
        {state.time_to_harm && (
          <div className="col-span-2">
            <p className="text-xs text-gray-500 mb-1">Time to Harm</p>
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-orange-100 text-orange-600">
              <Clock className="w-4 h-4" />
              <span className="font-semibold">{state.time_to_harm.hours.toFixed(1)} hrs</span>
              <span className="text-xs opacity-75">({state.time_to_harm.intervention_window})</span>
            </div>
          </div>
        )}
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

function AlertFeedComponent({ alerts, onAcknowledge }) {
  const [expandedId, setExpandedId] = useState(null);

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      <div className="p-3 border-b flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bell className="w-5 h-5 text-gray-500" />
          <h3 className="font-semibold">Alert Feed</h3>
          <span className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full">{alerts.length}</span>
        </div>
      </div>
      <div className="max-h-[500px] overflow-y-auto p-3 space-y-2">
        {alerts.length === 0 ? (
          <p className="text-center text-gray-500 py-8">No alerts to display</p>
        ) : (
          alerts.map(alert => {
            const severityConfig = SEVERITY_LEVELS[alert.severity] || SEVERITY_LEVELS.INFO;
            const domainConfig = RISK_DOMAINS.find(d => d.id === alert.risk_domain) || { name: alert.risk_domain, color: '#64748b', icon: '?' };
            const isInterruptive = alert.alert_type === 'interruptive';
            const isExpanded = expandedId === alert.event_id;

            return (
              <div key={alert.event_id} className={`bg-white rounded-lg shadow-sm border-l-4 p-3 ${isInterruptive ? 'animate-pulse-slow' : ''}`} style={{ borderLeftColor: severityConfig.color }}>
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold text-gray-900">{alert.patient_id}</span>
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded text-white" style={{ backgroundColor: domainConfig.color }}>
                        {domainConfig.icon} {domainConfig.name}
                      </span>
                      <span className="px-2 py-0.5 text-xs font-semibold rounded" style={{ backgroundColor: severityConfig.bgColor, color: severityConfig.color }}>
                        {alert.severity}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 truncate">{alert.clinical_headline || `State: ${alert.state_current}`}</p>
                    <p className="text-xs text-gray-400 mt-1">{new Date(alert.timestamp).toLocaleString()}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {isInterruptive && !alert.acknowledged && (
                      <button onClick={() => onAcknowledge(alert.event_id)} className="px-3 py-1 bg-blue-500 text-white text-sm rounded hover:bg-blue-600">ACK</button>
                    )}
                    <button onClick={() => setExpandedId(isExpanded ? null : alert.event_id)} className="p-1 hover:bg-gray-100 rounded">
                      {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
                {isExpanded && (
                  <div className="mt-3 pt-3 border-t text-sm space-y-2">
                    <div className="grid grid-cols-2 gap-2">
                      <div><span className="text-gray-500">State:</span> {alert.state_previous || '—'} → {alert.state_current}</div>
                      <div><span className="text-gray-500">Risk:</span> {((alert.risk_score || 0) * 100).toFixed(1)}%</div>
                    </div>
                    {alert.recommendations && (<ul className="list-disc list-inside text-gray-600">{alert.recommendations.slice(0, 3).map((r, i) => <li key={i}>{r}</li>)}</ul>)}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
