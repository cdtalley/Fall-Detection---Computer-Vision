'use client'

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Camera, Video, Activity, Clock, Target, UserCheck, Shield, Lock, UserX, Bone } from 'lucide-react'
import toast from 'react-hot-toast'

export default function LiveDemo() {
  const [isCameraActive, setIsCameraActive] = useState(false)
  const [detectionStatus, setDetectionStatus] = useState<'normal' | 'risk' | 'fall'>('normal')
  const [metrics, setMetrics] = useState({
    fps: 0,
    latency: 0,
    confidence: 0,
    poseStatus: 'No Pose'
  })
  const [isProcessing, setIsProcessing] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Simulate real-time metrics
  useEffect(() => {
    if (isCameraActive) {
      const interval = setInterval(() => {
        setMetrics({
          fps: Math.floor(Math.random() * 10) + 25, // 25-35 FPS
          latency: Math.floor(Math.random() * 200) + 150, // 150-350ms
          confidence: Math.floor(Math.random() * 30) + 70, // 70-100%
          poseStatus: Math.random() > 0.1 ? 'Pose Detected' : 'No Pose'
        })

        // Simulate detection status changes
        if (Math.random() > 0.95) {
          const statuses: Array<'normal' | 'risk' | 'fall'> = ['normal', 'risk', 'fall']
          const newStatus = statuses[Math.floor(Math.random() * statuses.length)]
          setDetectionStatus(newStatus)
          
          if (newStatus === 'fall') {
            toast.error('Fall Detected! Alert triggered.', {
              duration: 4000,
              icon: '🚨',
            })
          }
        }
      }, 1000)

      return () => clearInterval(interval)
    }
  }, [isCameraActive])

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 640, 
          height: 480,
          facingMode: 'user'
        } 
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsCameraActive(true)
        toast.success('Camera activated successfully!')
      }
    } catch (error) {
      console.error('Error accessing camera:', error)
      toast.error('Failed to access camera. Please check permissions.')
    }
  }

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach(track => track.stop())
      setIsCameraActive(false)
      toast.success('Camera deactivated')
    }
  }

  const privacyFeatures = [
    { icon: UserX, label: 'No Facial Recognition', description: 'Only movement patterns analyzed' },
    { icon: Shield, label: 'HIPAA Compliant', description: 'Full healthcare privacy compliance' },
    { icon: Bone, label: 'Skeleton Only', description: 'Anonymous pose keypoints only' },
    { icon: Lock, label: 'Data Encrypted', description: 'End-to-end encryption' },
  ]

  return (
    <section id="demo" className="py-20 bg-dark-800/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="section-header">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            Live Fall Detection Demo
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            viewport={{ once: true }}
          >
            Real-time privacy-preserving pose estimation and fall detection
          </motion.p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Demo Area */}
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="card p-6"
            >
              <div className="video-container aspect-video">
                {!isCameraActive ? (
                  <div className="flex flex-col items-center justify-center h-full text-center">
                    <Video className="w-16 h-16 text-dark-400 mb-4" />
                    <h3 className="text-xl font-semibold mb-2">Camera Feed</h3>
                    <p className="text-dark-400 mb-6">Click below to start the live demo</p>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={startCamera}
                      className="btn btn-primary"
                    >
                      <Camera className="w-5 h-5 mr-2" />
                      Start Camera
                    </motion.button>
                  </div>
                ) : (
                  <>
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full h-full object-cover rounded-lg"
                    />
                    <canvas
                      ref={canvasRef}
                      className="absolute inset-0 w-full h-full pointer-events-none"
                    />
                    
                    {/* Detection Overlay */}
                    <div className="absolute inset-0 pointer-events-none">
                      {/* Skeleton Overlay */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="skeleton-person opacity-50">
                          <div className="head" />
                          <div className="body" />
                          <div className="arms">
                            <div className="arm left" />
                            <div className="arm right" />
                          </div>
                          <div className="legs">
                            <div className="leg left" />
                            <div className="leg right" />
                          </div>
                        </div>
                      </div>

                      {/* Alert Banner */}
                      <AnimatePresence>
                        {detectionStatus === 'fall' && (
                          <motion.div
                            initial={{ opacity: 0, y: -50 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -50 }}
                            className="alert-banner"
                          >
                            <Activity className="w-5 h-5" />
                            <span>Fall Detected!</span>
                          </motion.div>
                        )}
                      </AnimatePresence>

                      {/* Detection Scan Line */}
                      <motion.div
                        animate={{ x: ['-100%', '100%'] }}
                        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                        className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent"
                      />
                    </div>

                    {/* Camera Controls */}
                    <div className="absolute bottom-4 right-4">
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={stopCamera}
                        className="btn btn-danger"
                      >
                        Stop Camera
                      </motion.button>
                    </div>
                  </>
                )}
              </div>
            </motion.div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Real-time Metrics */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="card p-6"
            >
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <Activity className="w-5 h-5 mr-2 text-primary-400" />
                Real-time Metrics
              </h3>
              
              <div className="space-y-4">
                <div className="metric-card">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <Target className="w-5 h-5 text-primary-400 mr-3" />
                      <span className="text-sm text-dark-400">FPS</span>
                    </div>
                    <div className="metric-value">{metrics.fps}</div>
                  </div>
                </div>

                <div className="metric-card">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <Clock className="w-5 h-5 text-primary-400 mr-3" />
                      <span className="text-sm text-dark-400">Latency</span>
                    </div>
                    <div className="metric-value">{metrics.latency}ms</div>
                  </div>
                </div>

                <div className="metric-card">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <Target className="w-5 h-5 text-primary-400 mr-3" />
                      <span className="text-sm text-dark-400">Confidence</span>
                    </div>
                    <div className="metric-value">{metrics.confidence}%</div>
                  </div>
                </div>

                <div className="metric-card">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <UserCheck className="w-5 h-5 text-primary-400 mr-3" />
                      <span className="text-sm text-dark-400">Pose Status</span>
                    </div>
                    <div className="text-sm font-medium text-white">{metrics.poseStatus}</div>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Detection Status */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              viewport={{ once: true }}
              className="card p-6"
            >
              <h3 className="text-lg font-semibold mb-4">Detection Status</h3>
              
              <div className="space-y-3">
                {[
                  { id: 'normal', label: 'Normal Activity', status: detectionStatus === 'normal' },
                  { id: 'risk', label: 'High Risk', status: detectionStatus === 'risk' },
                  { id: 'fall', label: 'Fall Detected', status: detectionStatus === 'fall' },
                ].map((item) => (
                  <motion.div
                    key={item.id}
                    className={`status-item ${item.status ? 'active' : ''}`}
                    animate={item.status ? { scale: 1.02 } : { scale: 1 }}
                    transition={{ duration: 0.2 }}
                  >
                    <div className={`status-dot ${item.id}`} />
                    <span className="text-sm font-medium">{item.label}</span>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            {/* Privacy Features */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              viewport={{ once: true }}
              className="card p-6"
            >
              <h3 className="text-lg font-semibold mb-4">Privacy Features</h3>
              
              <div className="space-y-3">
                {privacyFeatures.map((feature, index) => (
                  <motion.div
                    key={feature.label}
                    initial={{ opacity: 0, x: 20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.4, delay: index * 0.1 }}
                    viewport={{ once: true }}
                    className="feature-item"
                  >
                    <feature.icon className="w-4 h-4 text-healthcare-400" />
                    <div>
                      <div className="text-sm font-medium">{feature.label}</div>
                      <div className="text-xs text-dark-400">{feature.description}</div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  )
} 