'use client'

import { motion } from 'framer-motion'
import { Brain, Lock, Zap, Activity, Code, Database, Shield, Building2 } from 'lucide-react'

export default function AboutSection() {
  const features = [
    {
      icon: Brain,
      title: 'Advanced Computer Vision',
      description: 'Utilizes MediaPipe pose estimation for real-time skeleton tracking without facial recognition, ensuring complete privacy preservation.',
      color: 'primary'
    },
    {
      icon: Lock,
      title: 'Privacy-First Design',
      description: 'Processes only skeletal keypoints, never storing or transmitting video data. Compliant with HIPAA and GDPR requirements.',
      color: 'healthcare'
    },
    {
      icon: Zap,
      title: 'Real-time Processing',
      description: 'Sub-50ms inference times with optimized LSTM/CNN hybrid architecture for immediate fall detection and alert generation.',
      color: 'success'
    },
    {
      icon: Activity,
      title: 'Production-Ready Architecture',
      description: 'Built with scalability in mind, featuring comprehensive error handling, monitoring, and deployment-ready infrastructure.',
      color: 'warning'
    },
    {
      icon: Code,
      title: 'Open Source',
      description: 'Full source code available with detailed documentation, making it perfect for research, education, and commercial applications.',
      color: 'primary'
    },
    {
      icon: Database,
      title: 'Multi-Dataset Training',
      description: 'Trained on UP-Fall Detection, NTU RGB+D, and Kaggle datasets for robust performance across diverse scenarios.',
      color: 'healthcare'
    }
  ]

  const technicalSpecs = [
    { label: 'Framework', value: 'TensorFlow 2.x / PyTorch' },
    { label: 'Computer Vision', value: 'OpenCV, MediaPipe' },
    { label: 'Architecture', value: 'LSTM-CNN Hybrid' },
    { label: 'Processing', value: 'Real-time (30 FPS)' },
    { label: 'Privacy', value: 'Skeleton-only processing' },
    { label: 'Deployment', value: 'Docker, Kubernetes ready' },
  ]

  return (
    <section id="about" className="py-20 bg-dark-800/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            <span className="gradient-text">Technical</span>
            <span className="text-white"> Implementation</span>
          </h2>
          <p className="text-xl text-dark-300 max-w-3xl mx-auto">
            A comprehensive computer vision system demonstrating advanced AI/ML capabilities, 
            privacy-preserving design, and production-ready architecture for healthcare applications.
          </p>
        </motion.div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="bg-dark-700/50 backdrop-blur-sm rounded-xl p-6 border border-dark-600 hover:border-primary-500/30 transition-all duration-300"
            >
              <div className={`inline-flex p-3 rounded-lg bg-${feature.color}-500/10 border border-${feature.color}-500/20 mb-4`}>
                <feature.icon className={`w-6 h-6 text-${feature.color}-400`} />
              </div>
              <h3 className="text-xl font-semibold text-white mb-3">{feature.title}</h3>
              <p className="text-dark-300 leading-relaxed">{feature.description}</p>
            </motion.div>
          ))}
        </div>

        {/* Technical Specifications */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="bg-dark-700/30 backdrop-blur-sm rounded-2xl p-8 border border-dark-600"
        >
          <div className="text-center mb-8">
            <h3 className="text-2xl font-bold text-white mb-2">Technical Specifications</h3>
            <p className="text-dark-300">Core technologies and architecture details</p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {technicalSpecs.map((spec, index) => (
              <motion.div
                key={spec.label}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="flex justify-between items-center py-3 border-b border-dark-600 last:border-b-0"
              >
                <span className="text-dark-300 font-medium">{spec.label}</span>
                <span className="text-white font-semibold">{spec.value}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mt-12"
        >
          <div className="bg-gradient-to-r from-primary-500/10 to-healthcare-500/10 rounded-2xl p-8 border border-primary-500/20">
            <h3 className="text-2xl font-bold text-white mb-4">
              Ready to Explore the Code?
            </h3>
            <p className="text-dark-300 mb-6 max-w-2xl mx-auto">
              This project demonstrates comprehensive expertise in computer vision, machine learning, 
              privacy-preserving AI, and production-ready system architecture.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-6 py-3 bg-primary-500 text-white font-semibold rounded-lg hover:bg-primary-600 transition-colors duration-200"
              >
                View GitHub Repository
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-6 py-3 bg-dark-600 text-white font-semibold rounded-lg hover:bg-dark-500 transition-colors duration-200"
              >
                Download Dataset
              </motion.button>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
} 