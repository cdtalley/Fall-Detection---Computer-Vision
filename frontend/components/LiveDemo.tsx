'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Activity, Target, TrendingUp, AlertTriangle, CheckCircle, Clock, Zap, Brain, Shield, BarChart3, PieChart, LineChart } from 'lucide-react'

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

interface PerformanceData {
  timestamp: string
  accuracy: number
  latency: number
  confidence: number
}

export default function ModelPerformance() {
  const [selectedModel, setSelectedModel] = useState('lstm_cnn_hybrid')
  const [timeRange, setTimeRange] = useState('24h')
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

  const [performanceHistory, setPerformanceHistory] = useState<PerformanceData[]>([
    { timestamp: '00:00', accuracy: 93.1, latency: 0.9, confidence: 87.2 },
    { timestamp: '04:00', accuracy: 94.5, latency: 0.7, confidence: 89.1 },
    { timestamp: '08:00', accuracy: 95.2, latency: 0.8, confidence: 91.3 },
    { timestamp: '12:00', accuracy: 94.8, latency: 0.6, confidence: 90.7 },
    { timestamp: '16:00', accuracy: 95.1, latency: 0.8, confidence: 92.1 },
    { timestamp: '20:00', accuracy: 94.3, latency: 0.9, confidence: 88.9 },
    { timestamp: '24:00', accuracy: 94.2, latency: 0.8, confidence: 89.5 }
  ])

  const models = [
    { id: 'lstm_cnn_hybrid', name: 'LSTM-CNN Hybrid', description: 'Temporal + Spatial features' },
    { id: 'transformer', name: 'Vision Transformer', description: 'Attention-based architecture' },
    { id: 'efficientnet', name: 'EfficientNet-B4', description: 'Optimized for edge deployment' }
  ]

  const getStatusColor = (value: number, threshold: number) => {
    return value >= threshold ? 'text-green-400' : 'text-red-400'
  }

  return (
    <section id="model-performance" className="py-20 bg-dark-800/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="section-header">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            Model Performance Analysis
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            viewport={{ once: true }}
            className="text-lg text-dark-400 max-w-3xl mx-auto"
          >
            Comprehensive analysis of our fall detection model performance, showcasing accuracy, latency, and real-time metrics.
          </motion.p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Model Selection & Overview */}
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="card p-6"
            >
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
            </motion.div>
          </div>

          {/* Real-time Performance Panel */}
          <div className="space-y-6">
            {/* Current Performance */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="card p-6"
            >
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
            </motion.div>

            {/* Model Health */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              viewport={{ once: true }}
              className="card p-6"
            >
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
            </motion.div>

            {/* Technical Specifications */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              viewport={{ once: true }}
              className="card p-6"
            >
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
            </motion.div>
          </div>
        </div>

        {/* Performance History Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          viewport={{ once: true }}
          className="mt-8"
        >
          <div className="card p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <LineChart className="w-5 h-5 text-primary-400" />
              Performance History (24h)
            </h3>
            <div className="h-64 bg-dark-800 rounded-lg p-4">
              <div className="flex items-end justify-between h-full space-x-2">
                {performanceHistory.map((data, index) => (
                  <div key={index} className="flex flex-col items-center">
                    <div 
                      className="bg-gradient-to-t from-primary-500 to-primary-300 rounded-t w-8"
                      style={{ height: `${(data.accuracy / 100) * 200}px` }}
                    ></div>
                    <div className="text-xs text-dark-400 mt-2">{data.timestamp}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
} 