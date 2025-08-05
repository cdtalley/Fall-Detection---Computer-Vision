'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { BarChart, LineChart, PieChart, TrendingUp, CheckCircle, AlertTriangle, Clock, Shield } from 'lucide-react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js'
import { Line, Bar, Doughnut } from 'react-chartjs-2'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

export default function Analytics() {
  const [accuracyData, setAccuracyData] = useState({
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    datasets: [
      {
        label: 'Detection Accuracy',
        data: [89, 91, 93, 92, 94, 94.2],
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        fill: true,
        tension: 0.4,
      },
    ],
  })

  const [responseTimeData, setResponseTimeData] = useState({
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: [
      {
        label: 'Average Response Time (ms)',
        data: [320, 280, 250, 300, 240, 220, 247],
        backgroundColor: 'rgba(59, 130, 246, 0.8)',
        borderColor: 'rgb(59, 130, 246)',
        borderWidth: 2,
      },
    ],
  })

  const [complianceData, setComplianceData] = useState({
    labels: ['Sensitivity', 'Specificity', 'Precision', 'F1-Score'],
    datasets: [
      {
        data: [94.2, 87.3, 89.1, 91.5],
        backgroundColor: [
          'rgba(34, 197, 94, 0.8)',
          'rgba(59, 130, 246, 0.8)',
          'rgba(245, 158, 11, 0.8)',
          'rgba(168, 85, 247, 0.8)',
        ],
        borderColor: [
          'rgb(34, 197, 94)',
          'rgb(59, 130, 246)',
          'rgb(245, 158, 11)',
          'rgb(168, 85, 247)',
        ],
        borderWidth: 2,
      },
    ],
  })

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: '#f8fafc',
        },
      },
    },
    scales: {
      x: {
        ticks: {
          color: '#94a3b8',
        },
        grid: {
          color: 'rgba(148, 163, 184, 0.1)',
        },
      },
      y: {
        ticks: {
          color: '#94a3b8',
        },
        grid: {
          color: 'rgba(148, 163, 184, 0.1)',
        },
      },
    },
  }

  const complianceItems = [
    {
      label: 'Sensitivity',
      value: '94.2%',
      status: 'success',
      description: 'Critical - don\'t miss real falls',
    },
    {
      label: 'False Positive Rate',
      value: '8.7%',
      status: 'success',
      description: 'Below 10% threshold',
    },
    {
      label: 'Response Time',
      value: '<500ms',
      status: 'success',
      description: 'Real-time response',
    },
    {
      label: 'Privacy Protection',
      value: '100%',
      status: 'success',
      description: 'Pose estimation only',
    },
  ]

  const alertHistory = [
    { time: '2:34 PM', type: 'Normal Activity', confidence: '98%', status: 'normal' },
    { time: '2:31 PM', type: 'High Risk Movement', confidence: '87%', status: 'risk' },
    { time: '2:28 PM', type: 'Normal Activity', confidence: '95%', status: 'normal' },
    { time: '2:25 PM', type: 'Fall Detected', confidence: '92%', status: 'fall' },
    { time: '2:22 PM', type: 'Normal Activity', confidence: '97%', status: 'normal' },
    { time: '2:19 PM', type: 'High Risk Movement', confidence: '89%', status: 'risk' },
  ]

  return (
    <section id="analytics" className="py-20 bg-dark-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="section-header">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            Performance Analytics
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            viewport={{ once: true }}
          >
            Real-time system performance and healthcare compliance metrics
          </motion.p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Detection Accuracy Chart */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="card p-6"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold flex items-center">
                <TrendingUp className="w-5 h-5 mr-2 text-primary-400" />
                Detection Accuracy
              </h3>
            </div>
            <div className="chart-container">
              <Line data={accuracyData} options={chartOptions} />
            </div>
          </motion.div>

          {/* Response Time Chart */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            viewport={{ once: true }}
            className="card p-6"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold flex items-center">
                <Clock className="w-5 h-5 mr-2 text-primary-400" />
                Response Time
              </h3>
            </div>
            <div className="chart-container">
              <Bar data={responseTimeData} options={chartOptions} />
            </div>
          </motion.div>

          {/* Healthcare Compliance */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            viewport={{ once: true }}
            className="card p-6"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold flex items-center">
                <Shield className="w-5 h-5 mr-2 text-primary-400" />
                Healthcare Compliance
              </h3>
            </div>
            <div className="compliance-grid">
              {complianceItems.map((item, index) => (
                <motion.div
                  key={item.label}
                  initial={{ opacity: 0, x: 20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="compliance-item"
                >
                  <div className="compliance-icon success">
                    <CheckCircle className="w-4 h-4" />
                  </div>
                  <div className="compliance-content">
                    <div className="compliance-label">{item.label}</div>
                    <div className="compliance-value">{item.value}</div>
                    <div className="text-xs text-dark-400 mt-1">{item.description}</div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Alert History */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            viewport={{ once: true }}
            className="card p-6"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold flex items-center">
                <AlertTriangle className="w-5 h-5 mr-2 text-primary-400" />
                Alert History
              </h3>
            </div>
            <div className="alert-history">
              {alertHistory.map((alert, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className={`alert-item ${alert.status}`}
                >
                  <div className="alert-time">{alert.time}</div>
                  <div className="alert-type">{alert.type}</div>
                  <div className="alert-confidence">{alert.confidence}</div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Performance Metrics Summary */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          viewport={{ once: true }}
          className="mt-12 grid md:grid-cols-4 gap-6"
        >
          <div className="card p-6 text-center">
            <div className="text-3xl font-bold gradient-text mb-2">94.2%</div>
            <div className="text-sm text-dark-400">Overall Accuracy</div>
          </div>
          <div className="card p-6 text-center">
            <div className="text-3xl font-bold gradient-text mb-2">&lt;500ms</div>
            <div className="text-sm text-dark-400">Average Response</div>
          </div>
          <div className="card p-6 text-center">
            <div className="text-3xl font-bold gradient-text mb-2">99.9%</div>
            <div className="text-sm text-dark-400">System Uptime</div>
          </div>
          <div className="card p-6 text-center">
            <div className="text-3xl font-bold gradient-text mb-2">100%</div>
            <div className="text-sm text-dark-400">Privacy Safe</div>
          </div>
        </motion.div>
      </div>
    </section>
  )
} 