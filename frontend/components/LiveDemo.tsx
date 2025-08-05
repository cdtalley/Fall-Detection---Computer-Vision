'use client'

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Pause, RotateCcw, Video, Activity, Clock, Target, UserCheck, Shield, Lock, UserX, Bone, AlertTriangle, CheckCircle } from 'lucide-react'
import toast from 'react-hot-toast'

interface DetectionResult {
  timestamp: number
  confidence: number
  status: 'normal' | 'risk' | 'fall'
  pose_keypoints: [number, number][]
  bounding_box: [number, number, number, number]
  velocity: number
  stability_score: number
}

interface DemoVideo {
  id: string
  title: string
  description: string
  duration: number
  scenarios: Array<{
    time: number
    status: 'normal' | 'risk' | 'fall'
    description: string
  }>
  metrics: {
    accuracy: number
    sensitivity: number
    specificity: number
    response_time: number
  }
}

export default function LiveDemo() {
  const [currentVideo, setCurrentVideo] = useState<DemoVideo | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [detectionStatus, setDetectionStatus] = useState<'normal' | 'risk' | 'fall'>('normal')
  const [metrics, setMetrics] = useState({
    fps: 30,
    latency: 150,
    confidence: 0,
    poseStatus: 'No Pose'
  })
  const [showSkeleton, setShowSkeleton] = useState(true)
  const [showBoundingBox, setShowBoundingBox] = useState(true)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()

  // Demo videos data
  const demoVideos: DemoVideo[] = [
    {
      id: 'demo_1',
      title: 'Elderly Care Facility - Fall Detection',
      description: 'Real-time fall detection in a healthcare environment',
      duration: 30,
      scenarios: [
        { time: 0, status: 'normal', description: 'Normal walking' },
        { time: 10, status: 'risk', description: 'Unstable movement detected' },
        { time: 15, status: 'fall', description: 'Fall incident detected' },
        { time: 20, status: 'normal', description: 'Recovery and assistance' },
      ],
      metrics: {
        accuracy: 94.2,
        sensitivity: 96.8,
        specificity: 91.5,
        response_time: 0.8
      }
    },
    {
      id: 'demo_2',
      title: 'Home Safety Monitoring',
      description: 'Privacy-preserving fall detection in residential setting',
      duration: 45,
      scenarios: [
        { time: 0, status: 'normal', description: 'Daily activities' },
        { time: 20, status: 'risk', description: 'Potential fall risk' },
        { time: 35, status: 'fall', description: 'Fall detected' },
        { time: 40, status: 'normal', description: 'Emergency response' },
      ],
      metrics: {
        accuracy: 92.1,
        sensitivity: 94.3,
        specificity: 89.7,
        response_time: 1.2
      }
    }
  ]

  // Set default video
  useEffect(() => {
    if (!currentVideo && demoVideos.length > 0) {
      setCurrentVideo(demoVideos[0])
    }
  }, [currentVideo])

  // Simulated detection data based on video scenarios
  const getDetectionAtTime = (time: number): DetectionResult | null => {
    if (!currentVideo) return null

    // Find the scenario that applies to this time
    let currentScenario = currentVideo.scenarios[0]
    for (const scenario of currentVideo.scenarios) {
      if (time >= scenario.time) {
        currentScenario = scenario
      }
    }

    // Generate realistic detection data
    const confidence = currentScenario.status === 'normal' 
      ? Math.random() * 10 + 85  // 85-95%
      : currentScenario.status === 'risk'
      ? Math.random() * 15 + 70  // 70-85%
      : Math.random() * 8 + 90   // 90-98%

    return {
      timestamp: time,
      confidence,
      status: currentScenario.status,
      pose_keypoints: generatePoseKeypoints(currentScenario.status),
      bounding_box: generateBoundingBox(currentScenario.status),
      velocity: generateVelocity(currentScenario.status),
      stability_score: generateStability(currentScenario.status)
    }
  }

  const generatePoseKeypoints = (status: string): [number, number][] => {
    if (status === 'normal') {
      return [
        [0.5, 0.1], [0.48, 0.15], [0.52, 0.15], [0.45, 0.2], [0.55, 0.2],
        [0.5, 0.25], [0.5, 0.25], [0.45, 0.35], [0.55, 0.35], [0.4, 0.45],
        [0.6, 0.45], [0.5, 0.4], [0.5, 0.4], [0.45, 0.6], [0.55, 0.6],
        [0.4, 0.8], [0.6, 0.8]
      ]
    } else if (status === 'risk') {
      return [
        [0.5, 0.1], [0.48, 0.15], [0.52, 0.15], [0.45, 0.2], [0.55, 0.2],
        [0.48, 0.25], [0.52, 0.25], [0.43, 0.35], [0.57, 0.35], [0.38, 0.45],
        [0.62, 0.45], [0.48, 0.4], [0.52, 0.4], [0.43, 0.6], [0.57, 0.6],
        [0.35, 0.8], [0.65, 0.8]
      ]
    } else {
      return [
        [0.5, 0.3], [0.48, 0.32], [0.52, 0.32], [0.45, 0.34], [0.55, 0.34],
        [0.48, 0.4], [0.52, 0.4], [0.43, 0.5], [0.57, 0.5], [0.38, 0.6],
        [0.62, 0.6], [0.48, 0.6], [0.52, 0.6], [0.43, 0.75], [0.57, 0.75],
        [0.35, 0.9], [0.65, 0.9]
      ]
    }
  }

  const generateBoundingBox = (status: string): [number, number, number, number] => {
    if (status === 'normal') return [200, 100, 400, 600]
    if (status === 'risk') return [180, 120, 420, 580]
    return [150, 200, 450, 500]
  }

  const generateVelocity = (status: string): number => {
    if (status === 'normal') return Math.random() * 0.4 + 0.1
    if (status === 'risk') return Math.random() * 0.7 + 0.8
    return Math.random() * 2.0 + 2.0
  }

  const generateStability = (status: string): number => {
    if (status === 'normal') return Math.random() * 0.2 + 0.8
    if (status === 'risk') return Math.random() * 0.3 + 0.4
    return Math.random() * 0.3
  }

  const drawSkeleton = (ctx: CanvasRenderingContext2D, keypoints: [number, number][]) => {
    if (!showSkeleton) return

    const canvas = ctx.canvas
    const width = canvas.width
    const height = canvas.height

    // Draw keypoints
    ctx.strokeStyle = '#22c55e'
    ctx.lineWidth = 2
    ctx.fillStyle = '#22c55e'

    keypoints.forEach(([x, y], index) => {
      const pixelX = x * width
      const pixelY = y * height
      
      ctx.beginPath()
      ctx.arc(pixelX, pixelY, 4, 0, Math.PI * 2)
      ctx.fill()
    })

    // Draw skeleton connections
    const connections = [
      [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
      [5, 11], [6, 12], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]
    ]

    connections.forEach(([start, end]) => {
      if (start < keypoints.length && end < keypoints.length) {
        const [x1, y1] = keypoints[start]
        const [x2, y2] = keypoints[end]
        
        ctx.beginPath()
        ctx.moveTo(x1 * width, y1 * height)
        ctx.lineTo(x2 * width, y2 * height)
        ctx.stroke()
      }
    })
  }

  const drawBoundingBox = (ctx: CanvasRenderingContext2D, bbox: [number, number, number, number]) => {
    if (!showBoundingBox) return

    const [x, y, w, h] = bbox
    const color = detectionStatus === 'normal' ? '#22c55e' : 
                  detectionStatus === 'risk' ? '#f59e0b' : '#ef4444'
    
    ctx.strokeStyle = color
    ctx.lineWidth = 3
    ctx.setLineDash([5, 5])
    ctx.strokeRect(x, y, w, h)
    ctx.setLineDash([])
  }

  const updateCanvas = () => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Get current detection
    const detection = getDetectionAtTime(currentTime)
    if (detection) {
      setDetectionStatus(detection.status)
      setMetrics({
        fps: 30,
        latency: Math.random() * 60 + 120,
        confidence: detection.confidence,
        poseStatus: 'Pose Detected'
      })

      // Draw skeleton and bounding box
      drawSkeleton(ctx, detection.pose_keypoints)
      drawBoundingBox(ctx, detection.bounding_box)

      // Show fall alert
      if (detection.status === 'fall') {
        toast.error('Fall Detected! Alert triggered.', {
          duration: 4000,
          icon: '🚨',
        })
      }
    }
  }

  useEffect(() => {
    if (isPlaying) {
      const animate = () => {
        setCurrentTime(prev => {
          const newTime = prev + 0.1
          if (newTime >= (currentVideo?.duration || 30)) {
            setIsPlaying(false)
            return 0
          }
          return newTime
        })
        updateCanvas()
        animationRef.current = requestAnimationFrame(animate)
      }
      animationRef.current = requestAnimationFrame(animate)
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isPlaying, currentVideo])

  useEffect(() => {
    updateCanvas()
  }, [currentTime, showSkeleton, showBoundingBox])

  // Initialize demo when component mounts
  useEffect(() => {
    if (currentVideo) {
      // Start with some initial detection data
      const initialDetection = getDetectionAtTime(0)
      if (initialDetection) {
        setDetectionStatus(initialDetection.status)
        setMetrics({
          fps: 30,
          latency: 150,
          confidence: initialDetection.confidence,
          poseStatus: 'Pose Detected'
        })
      }
      updateCanvas()
    }
  }, [currentVideo])

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying)
  }

  const handleReset = () => {
    setCurrentTime(0)
    setIsPlaying(false)
  }

  const handleVideoSelect = (video: DemoVideo) => {
    setCurrentVideo(video)
    setCurrentTime(0)
    setIsPlaying(false)
  }

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
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
            className="text-lg text-dark-400 max-w-3xl mx-auto"
          >
            Watch our AI system in action as it detects falls in real-time using privacy-preserving computer vision technology.
          </motion.p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Video Player */}
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="card p-6"
            >
              <div className="mb-4">
                <h3 className="text-xl font-semibold mb-2">Demo Videos</h3>
                <div className="flex gap-2 mb-4">
                  {demoVideos.map((video) => (
                    <button
                      key={video.id}
                      onClick={() => handleVideoSelect(video)}
                      className={`px-3 py-1 rounded text-sm transition-colors ${
                        currentVideo?.id === video.id
                          ? 'bg-primary-600 text-white'
                          : 'bg-dark-700 text-dark-300 hover:bg-dark-600'
                      }`}
                    >
                      {video.title.split(' - ')[0]}
                    </button>
                  ))}
                </div>
              </div>

              {currentVideo ? (
                <>
                  <div className="video-container h-96 mb-4">
                    <div className="relative w-full h-full bg-dark-900 rounded-lg overflow-hidden border-2 border-dashed border-primary-500/30">
                      {/* Placeholder for video - in real implementation, this would be an actual video */}
                      <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-dark-800 to-dark-900">
                        <div className="text-center">
                          <Video className="w-16 h-16 text-primary-500 mx-auto mb-4" />
                          <h4 className="text-lg font-semibold mb-2">{currentVideo.title}</h4>
                          <p className="text-dark-400 text-sm mb-4">{currentVideo.description}</p>
                          <div className="text-xs text-dark-500">
                            Duration: {formatTime(currentVideo.duration)} | 
                            Accuracy: {currentVideo.metrics.accuracy}%
                          </div>
                        </div>
                      </div>
                      
                      {/* Detection overlay canvas */}
                      <canvas
                        ref={canvasRef}
                        className="absolute inset-0 pointer-events-none"
                        width={640}
                        height={480}
                      />
                      
                      {/* Status indicator */}
                      <div className="absolute top-4 left-4">
                        <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium ${
                          detectionStatus === 'normal' ? 'bg-healthcare-500/20 text-healthcare-400' :
                          detectionStatus === 'risk' ? 'bg-warning-500/20 text-warning-400' :
                          'bg-danger-500/20 text-danger-400'
                        }`}>
                          <div className={`w-2 h-2 rounded-full ${
                            detectionStatus === 'normal' ? 'bg-healthcare-400' :
                            detectionStatus === 'risk' ? 'bg-warning-400' :
                            'bg-danger-400'
                          }`} />
                          {detectionStatus === 'normal' ? 'Normal' :
                           detectionStatus === 'risk' ? 'Risk Detected' : 'Fall Detected'}
                        </div>
                      </div>

                      {/* Fall alert banner */}
                      <AnimatePresence>
                        {detectionStatus === 'fall' && (
                          <motion.div
                            initial={{ opacity: 0, y: -20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            className="alert-banner"
                          >
                            <AlertTriangle className="w-5 h-5" />
                            FALL DETECTED - Emergency Alert Sent
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </div>

                  {/* Video controls */}
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-4">
                      <button
                        onClick={handlePlayPause}
                        className="btn btn-primary"
                      >
                        {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                        {isPlaying ? 'Pause' : 'Play'}
                      </button>
                      <button
                        onClick={handleReset}
                        className="btn btn-secondary"
                      >
                        <RotateCcw className="w-4 h-4" />
                        Reset
                      </button>
                    </div>
                    <div className="text-sm text-dark-400">
                      {formatTime(currentTime)} / {formatTime(currentVideo.duration)}
                    </div>
                  </div>

                  {/* Progress bar */}
                  <div className="w-full bg-dark-700 rounded-full h-2 mb-4">
                    <div
                      className="bg-gradient-to-r from-primary-500 to-healthcare-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${(currentTime / currentVideo.duration) * 100}%` }}
                    />
                  </div>

                  {/* Detection controls */}
                  <div className="flex items-center gap-4 text-sm">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={showSkeleton}
                        onChange={(e) => setShowSkeleton(e.target.checked)}
                        className="rounded"
                      />
                      Show Skeleton
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={showBoundingBox}
                        onChange={(e) => setShowBoundingBox(e.target.checked)}
                        className="rounded"
                      />
                      Show Bounding Box
                    </label>
                  </div>
                </>
              ) : (
                <div className="video-container h-96 mb-4">
                  <div className="relative w-full h-full bg-dark-900 rounded-lg overflow-hidden">
                    <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-dark-800 to-dark-900">
                      <div className="text-center">
                        <Video className="w-16 h-16 text-primary-500 mx-auto mb-4 animate-pulse" />
                        <h4 className="text-lg font-semibold mb-2">Select a Demo Video</h4>
                        <p className="text-dark-400 text-sm mb-4">Choose from the options above to start the fall detection demo</p>
                        <div className="text-xs text-dark-500">
                          Click on "Elderly Care Facility" or "Home Safety Monitoring"
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          </div>

          {/* Metrics Panel */}
          <div className="space-y-6">
            {/* Real-time Metrics */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="card p-6"
            >
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-primary-400" />
                Real-time Metrics
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-dark-400">FPS</span>
                  <span className="font-mono text-primary-400">{metrics.fps}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-dark-400">Latency</span>
                  <span className="font-mono text-primary-400">{Math.round(metrics.latency)}ms</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-dark-400">Confidence</span>
                  <span className="font-mono text-primary-400">{Math.round(metrics.confidence)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-dark-400">Pose Status</span>
                  <span className="font-mono text-primary-400">{metrics.poseStatus}</span>
                </div>
              </div>
            </motion.div>

            {/* Current Video Info */}
            {currentVideo ? (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
                viewport={{ once: true }}
                className="card p-6"
              >
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Video className="w-5 h-5 text-primary-400" />
                  Video Analysis
                </h3>
                <div className="space-y-3">
                  <div>
                    <div className="text-sm text-dark-400">Accuracy</div>
                    <div className="text-lg font-semibold text-healthcare-400">{currentVideo.metrics.accuracy}%</div>
                  </div>
                  <div>
                    <div className="text-sm text-dark-400">Sensitivity</div>
                    <div className="text-lg font-semibold text-primary-400">{currentVideo.metrics.sensitivity}%</div>
                  </div>
                  <div>
                    <div className="text-sm text-dark-400">Response Time</div>
                    <div className="text-lg font-semibold text-warning-400">{currentVideo.metrics.response_time}s</div>
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
                viewport={{ once: true }}
                className="card p-6"
              >
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Video className="w-5 h-5 text-primary-400" />
                  Demo Status
                </h3>
                <div className="space-y-3">
                  <div className="text-center text-dark-400">
                    <p className="text-sm">No video selected</p>
                    <p className="text-xs mt-1">Select a demo video to see analysis</p>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Privacy Features */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              viewport={{ once: true }}
              className="card p-6"
            >
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Shield className="w-5 h-5 text-primary-400" />
                Privacy Features
              </h3>
              <div className="space-y-3">
                {privacyFeatures.map((feature, index) => (
                  <div key={index} className="flex items-start gap-3">
                    <feature.icon className="w-4 h-4 text-healthcare-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <div className="text-sm font-medium">{feature.label}</div>
                      <div className="text-xs text-dark-400">{feature.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  )
} 