'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, Target, TrendingUp, AlertTriangle, CheckCircle, Brain, BarChart3, Settings, Filter, Eye, LineChart } from 'lucide-react'

interface ModelMetrics {
  accuracy: number
  precision: number
  recall: number
  f1Score: number
  sensitivity: number
  specificity: number
  latency: number
  throughput: number
  falsePositiveRate: number
  falseNegativeRate: number
}

interface FalseDetectionCase {
  id: string
  type: 'false_positive' | 'false_negative'
  confidence: number
  timestamp: string
  scenario: string
  impact: 'high' | 'medium' | 'low'
  suggestedImprovement: string
  features: string[]
}

interface ModelImprovement {
  strategy: string
  description: string
  expectedImprovement: string
  implementation: string
  priority: 'high' | 'medium' | 'low'
  status: 'implemented' | 'in_progress' | 'planned'
}

export default function ModelPerformance() {
  const [selectedModel, setSelectedModel] = useState('lstm_cnn_hybrid')
  const [activeTab, setActiveTab] = useState('overview')
  
  const [metrics, setMetrics] = useState<ModelMetrics>({
    accuracy: 94.2,
    precision: 92.8,
    recall: 96.5,
    f1Score: 94.6,
    sensitivity: 96.8,
    specificity: 91.5,
    latency: 0.8,
    throughput: 1250,
    falsePositiveRate: 8.5,
    falseNegativeRate: 3.2
  })

  const [falseDetectionCases, setFalseDetectionCases] = useState<FalseDetectionCase[]>([
    {
      id: 'fp_001',
      type: 'false_positive',
      confidence: 87.3,
      timestamp: '2024-01-15 14:23',
      scenario: 'Person bending down to pick up object',
      impact: 'high',
      suggestedImprovement: 'Add temporal consistency check for fall-like movements',
      features: ['rapid_descent', 'ground_proximity', 'lack_of_recovery']
    },
    {
      id: 'fn_001',
      type: 'false_negative',
      confidence: 45.2,
      timestamp: '2024-01-15 16:47',
      scenario: 'Slow fall with partial occlusion',
      impact: 'high',
      suggestedImprovement: 'Implement multi-frame analysis and occlusion handling',
      features: ['slow_movement', 'partial_occlusion', 'low_confidence']
    }
  ])

  const [modelImprovements, setModelImprovements] = useState<ModelImprovement[]>([
    {
      strategy: 'Temporal Consistency Validation',
      description: 'Implement multi-frame analysis to distinguish between falls and similar movements',
      expectedImprovement: 'Reduce false positives by 15-20%',
      implementation: 'Add LSTM layer with 10-frame sequence analysis',
      priority: 'high',
      status: 'implemented'
    },
    {
      strategy: 'Confidence Threshold Optimization',
      description: 'Dynamic threshold adjustment based on environmental conditions',
      expectedImprovement: 'Improve precision by 8-12%',
      implementation: 'Bayesian optimization for threshold tuning',
      priority: 'high',
      status: 'in_progress'
    }
  ])

  const models = [
    { id: 'lstm_cnn_hybrid', name: 'LSTM-CNN Hybrid', description: 'Temporal + Spatial features' },
    { id: 'transformer', name: 'Vision Transformer', description: 'Attention-based architecture' },
    { id: 'efficientnet', name: 'EfficientNet-B4', description: 'Optimized for edge deployment' }
  ]

  const getStatusColor = (value: number, threshold: number) => {
    return value >= threshold ? 'text-green-400' : 'text-red-400'
  }

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'text-red-400'
      case 'medium': return 'text-yellow-400'
      case 'low': return 'text-green-400'
      default: return 'text-gray-400'
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      case 'low': return 'bg-green-500/20 text-green-400 border-green-500/30'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  const getStatusColor2 = (status: string) => {
    switch (status) {
      case 'implemented': return 'text-green-400'
      case 'in_progress': return 'text-blue-400'
      case 'planned': return 'text-yellow-400'
      default: return 'text-gray-400'
    }
  }

  return (
    <div className="py-20 bg-dark-800/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold mb-4">
            Advanced Model Analysis & Improvement
          </h2>
          <p className="text-lg text-dark-400 max-w-4xl mx-auto">
            Comprehensive analysis focusing on false positive/negative detection, precision/recall optimization, 
            and continuous model improvement strategies for healthcare applications.
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex flex-wrap gap-2 mb-8 justify-center">
          {[
            { id: 'overview', label: 'Performance Overview', icon: BarChart3 },
            { id: 'false_detections', label: 'False Detection Analysis', icon: AlertTriangle },
            { id: 'improvements', label: 'Model Improvements', icon: Settings },
            { id: 'optimization', label: 'Precision/Recall Optimization', icon: Target }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-colors ${
                activeTab === tab.id
                  ? 'bg-primary-600 text-white'
                  : 'bg-dark-700 text-dark-300 hover:bg-dark-600'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {activeTab === 'overview' && (
          <div className="grid lg:grid-cols-3 gap-8">
            {/* Model Selection & Overview */}
            <div className="lg:col-span-2">
              <div className="bg-dark-700/50 backdrop-blur-sm rounded-xl p-6 border border-dark-600">
                <div className="mb-6">
                  <h3 className="text-xl font-semibold mb-4">Model Architecture</h3>
                  <div className="flex gap-2 mb-4">
                    {models.map((model) => (
                      <button
                        key={model.id}
                        onClick={() => setSelectedModel(model.id)}
                        className={`px-4 py-2 rounded-lg text-sm transition-colors ${
                          selectedModel === model.id
                            ? 'bg-primary-600 text-white'
                            : 'bg-dark-700 text-dark-300 hover:bg-dark-600'
                        }`}
                      >
                        {model.name}
                      </button>
                    ))}
                  </div>
                  <div className="bg-dark-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-2">
                      {models.find(m => m.id === selectedModel)?.name}
                    </h4>
                    <p className="text-dark-400 text-sm">
                      {models.find(m => m.id === selectedModel)?.description}
                    </p>
                  </div>
                </div>

                {/* Performance Metrics Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="bg-dark-800 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-green-400 mb-1">
                      {metrics.accuracy}%
                    </div>
                    <div className="text-xs text-dark-400">Accuracy</div>
                  </div>
                  <div className="bg-dark-800 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-blue-400 mb-1">
                      {metrics.f1Score}%
                    </div>
                    <div className="text-xs text-dark-400">F1 Score</div>
                  </div>
                  <div className="bg-dark-800 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-yellow-400 mb-1">
                      {metrics.latency}s
                    </div>
                    <div className="text-xs text-dark-400">Latency</div>
                  </div>
                  <div className="bg-dark-800 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-purple-400 mb-1">
                      {metrics.throughput}
                    </div>
                    <div className="text-xs text-dark-400">FPS</div>
                  </div>
                </div>

                {/* Detailed Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold mb-4 flex items-center gap-2">
                      <Target className="w-4 h-4" />
                      Classification Metrics
                    </h4>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-dark-400">Precision</span>
                        <span className={`font-mono ${getStatusColor(metrics.precision, 90)}`}>
                          {metrics.precision}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-dark-400">Recall</span>
                        <span className={`font-mono ${getStatusColor(metrics.recall, 90)}`}>
                          {metrics.recall}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-dark-400">Sensitivity</span>
                        <span className={`font-mono ${getStatusColor(metrics.sensitivity, 90)}`}>
                          {metrics.sensitivity}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-dark-400">Specificity</span>
                        <span className={`font-mono ${getStatusColor(metrics.specificity, 85)}`}>
                          {metrics.specificity}%
                        </span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-4 flex items-center gap-2">
                      <Activity className="w-4 h-4" />
                      Error Analysis
                    </h4>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-dark-400">False Positive Rate</span>
                        <span className="font-mono text-red-400">
                          {metrics.falsePositiveRate}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-dark-400">False Negative Rate</span>
                        <span className="font-mono text-red-400">
                          {metrics.falseNegativeRate}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-dark-400">Alert Fatigue Risk</span>
                        <span className="font-mono text-yellow-400">
                          Low
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-dark-400">Missed Falls Risk</span>
                        <span className="font-mono text-green-400">
                          Very Low
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Real-time Performance Panel */}
            <div className="space-y-6">
              {/* Current Performance */}
              <div className="bg-dark-700/50 backdrop-blur-sm rounded-xl p-6 border border-dark-600">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-primary-400" />
                  Real-time Performance
                </h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-dark-400">Current Accuracy</span>
                    <span className="font-mono text-green-400">94.2%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-dark-400">Inference Latency</span>
                    <span className="font-mono text-blue-400">0.8ms</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-dark-400">Confidence Score</span>
                    <span className="font-mono text-yellow-400">89.5%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-dark-400">Model Status</span>
                    <span className="font-mono text-green-400">Online</span>
                  </div>
                </div>
              </div>

              {/* Model Health */}
              <div className="bg-dark-700/50 backdrop-blur-sm rounded-xl p-6 border border-dark-600">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-primary-400" />
                  Model Health
                </h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Data Quality</span>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span className="text-xs text-green-400">Excellent</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Model Drift</span>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span className="text-xs text-green-400">Stable</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Performance</span>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span className="text-xs text-green-400">Optimal</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Resource Usage</span>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                      <span className="text-xs text-yellow-400">Normal</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Technical Specifications */}
              <div className="bg-dark-700/50 backdrop-blur-sm rounded-xl p-6 border border-dark-600">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Brain className="w-5 h-5 text-primary-400" />
                  Technical Specs
                </h3>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-dark-400">Framework</span>
                    <span>TensorFlow 2.12</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-dark-400">Model Size</span>
                    <span>45.2 MB</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-dark-400">Parameters</span>
                    <span>2.1M</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-dark-400">Input Resolution</span>
                    <span>640x480</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-dark-400">Batch Size</span>
                    <span>32</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-dark-400">Optimization</span>
                    <span>TensorRT</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* False Detection Analysis Tab */}
        {activeTab === 'false_detections' && (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid md:grid-cols-4 gap-6">
              <div className="bg-dark-700/50 backdrop-blur-sm rounded-xl p-6 text-center border border-dark-600">
                <div className="text-3xl font-bold text-red-400 mb-2">12</div>
                <div className="text-sm text-dark-400">False Positives</div>
                <div className="text-xs text-red-400 mt-1">Last 24h</div>
              </div>
              <div className="bg-dark-700/50 backdrop-blur-sm rounded-xl p-6 text-center border border-dark-600">
                <div className="text-3xl font-bold text-orange-400 mb-2">3</div>
                <div className="text-sm text-dark-400">False Negatives</div>
                <div className="text-xs text-orange-400 mt-1">Last 24h</div>
              </div>
              <div className="bg-dark-700/50 backdrop-blur-sm rounded-xl p-6 text-center border border-dark-600">
                <div className="text-3xl font-bold text-blue-400 mb-2">8.5%</div>
                <div className="text-sm text-dark-400">FP Rate</div>
                <div className="text-xs text-blue-400 mt-1">Target: &lt;5%</div>
              </div>
              <div className="bg-dark-700/50 backdrop-blur-sm rounded-xl p-6 text-center border border-dark-600">
                <div className="text-3xl font-bold text-green-400 mb-2">3.2%</div>
                <div className="text-sm text-dark-400">FN Rate</div>
                <div className="text-xs text-green-400 mt-1">Target: &lt;2%</div>
              </div>
            </div>

            {/* False Detection Cases */}
            <div className="bg-dark-700/50 backdrop-blur-sm rounded-xl p-6 border border-dark-600">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-semibold flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-red-400" />
                  False Detection Analysis
                </h3>
                <div className="flex gap-2">
                  <button className="px-3 py-1 bg-dark-700 rounded text-sm hover:bg-dark-600">
                    <Filter className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="space-y-4">
                {falseDetectionCases.map((case_, index) => (
                  <div
                    key={case_.id}
                    className="bg-dark-800 rounded-lg p-4 border-l-4 border-red-400"
                  >
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            case_.type === 'false_positive' 
                              ? 'bg-red-500/20 text-red-400' 
                              : 'bg-orange-500/20 text-orange-400'
                          }`}>
                            {case_.type === 'false_positive' ? 'False Positive' : 'False Negative'}
                          </span>
                          <span className={`text-sm font-medium ${getImpactColor(case_.impact)}`}>
                            {case_.impact.toUpperCase()} IMPACT
                          </span>
                        </div>
                        <h4 className="font-semibold text-white mb-1">{case_.scenario}</h4>
                        <p className="text-sm text-dark-400 mb-2">{case_.suggestedImprovement}</p>
                      </div>
                      <div className="text-right">
                        <div className="text-sm text-dark-400">{case_.timestamp}</div>
                        <div className="text-lg font-bold text-yellow-400">{case_.confidence}%</div>
                      </div>
                    </div>
                    
                    <div className="flex flex-wrap gap-2">
                      {case_.features.map((feature, idx) => (
                        <span key={idx} className="px-2 py-1 bg-dark-700 rounded text-xs text-dark-300">
                          {feature}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Model Improvements Tab */}
        {activeTab === 'improvements' && (
          <div className="space-y-6">
            {/* Improvement Strategies */}
            <div className="grid lg:grid-cols-2 gap-6">
              {modelImprovements.map((improvement, index) => (
                <div
                  key={improvement.strategy}
                  className="bg-dark-700/50 backdrop-blur-sm rounded-xl p-6 border border-dark-600"
                >
                  <div className="flex justify-between items-start mb-4">
                    <h3 className="text-lg font-semibold">{improvement.strategy}</h3>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getPriorityColor(improvement.priority)}`}>
                      {improvement.priority.toUpperCase()}
                    </span>
                  </div>
                  
                  <p className="text-dark-400 mb-4">{improvement.description}</p>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-dark-400">Expected Improvement:</span>
                      <span className="text-sm font-medium text-green-400">{improvement.expectedImprovement}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-dark-400">Implementation:</span>
                      <span className="text-sm font-medium text-blue-400">{improvement.implementation}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-dark-400">Status:</span>
                      <span className={`text-sm font-medium ${getStatusColor2(improvement.status)}`}>
                        {improvement.status.replace('_', ' ').toUpperCase()}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Precision/Recall Optimization Tab */}
        {activeTab === 'optimization' && (
          <div className="space-y-6">
            {/* Optimization Strategies */}
            <div className="grid lg:grid-cols-2 gap-6">
              <div className="bg-dark-700/50 backdrop-blur-sm rounded-xl p-6 border border-dark-600">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Target className="w-5 h-5 text-primary-400" />
                  Precision Optimization
                </h3>
                <div className="space-y-4">
                  <div className="bg-dark-800 rounded-lg p-4">
                    <h4 className="font-medium mb-2">Confidence Threshold Tuning</h4>
                    <p className="text-sm text-dark-400 mb-3">Adjust decision threshold to reduce false positives</p>
                    <div className="flex items-center gap-4">
                      <div className="flex-1">
                        <div className="flex justify-between text-xs text-dark-400 mb-1">
                          <span>Current: 0.75</span>
                          <span>Optimal: 0.82</span>
                        </div>
                        <div className="w-full bg-dark-700 rounded-full h-2">
                          <div className="bg-blue-500 h-2 rounded-full" style={{ width: '75%' }}></div>
                        </div>
                      </div>
                      <button className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600">
                        Apply
                      </button>
                    </div>
                  </div>
                  
                  <div className="bg-dark-800 rounded-lg p-4">
                    <h4 className="font-medium mb-2">Temporal Consistency Check</h4>
                    <p className="text-sm text-dark-400 mb-3">Multi-frame validation for fall-like movements</p>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-dark-400">Implementation Status</span>
                      <span className="text-sm text-green-400">Active</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-dark-700/50 backdrop-blur-sm rounded-xl p-6 border border-dark-600">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Eye className="w-5 h-5 text-primary-400" />
                  Recall Optimization
                </h3>
                <div className="space-y-4">
                  <div className="bg-dark-800 rounded-lg p-4">
                    <h4 className="font-medium mb-2">Multi-Scale Detection</h4>
                    <p className="text-sm text-dark-400 mb-3">Detect falls at various distances and scales</p>
                    <div className="flex items-center gap-4">
                      <div className="flex-1">
                        <div className="flex justify-between text-xs text-dark-400 mb-1">
                          <span>Scales: 3</span>
                          <span>Target: 5</span>
                        </div>
                        <div className="w-full bg-dark-700 rounded-full h-2">
                          <div className="bg-green-500 h-2 rounded-full" style={{ width: '60%' }}></div>
                        </div>
                      </div>
                      <button className="px-3 py-1 bg-green-500 text-white rounded text-sm hover:bg-green-600">
                        Expand
                      </button>
                    </div>
                  </div>
                  
                  <div className="bg-dark-800 rounded-lg p-4">
                    <h4 className="font-medium mb-2">Occlusion Handling</h4>
                    <p className="text-sm text-dark-400 mb-3">Improved detection under partial occlusion</p>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-dark-400">Implementation Status</span>
                      <span className="text-sm text-yellow-400">In Progress</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Precision-Recall Curve */}
            <div className="bg-dark-700/50 backdrop-blur-sm rounded-xl p-6 border border-dark-600">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <LineChart className="w-5 h-5 text-primary-400" />
                Precision-Recall Curve Analysis
              </h3>
              <div className="h-64 bg-dark-800 rounded-lg p-4">
                <div className="flex items-end justify-between h-full space-x-2">
                  {[
                    { precision: 0.85, recall: 0.92, threshold: 0.6 },
                    { precision: 0.88, recall: 0.89, threshold: 0.7 },
                    { precision: 0.91, recall: 0.85, threshold: 0.8 },
                    { precision: 0.94, recall: 0.78, threshold: 0.9 },
                    { precision: 0.96, recall: 0.65, threshold: 0.95 }
                  ].map((point, index) => (
                    <div key={index} className="flex flex-col items-center">
                      <div className="text-xs text-dark-400 mb-1">{point.threshold}</div>
                      <div 
                        className="bg-gradient-to-t from-primary-500 to-primary-300 rounded-t w-8"
                        style={{ height: `${point.precision * 200}px` }}
                      ></div>
                      <div className="text-xs text-dark-400 mt-1">
                        P: {(point.precision * 100).toFixed(0)}%<br/>
                        R: {(point.recall * 100).toFixed(0)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
} 