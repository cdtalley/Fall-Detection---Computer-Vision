'use client'

import { motion } from 'framer-motion'
import { Brain, Shield, Zap, Building2, Lock, Activity, Target } from 'lucide-react'

export default function AboutSection() {
  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Detection',
      description: 'Advanced machine learning models using LSTM and CNN architectures for accurate fall detection with 94.2% sensitivity.',
      color: 'text-primary-400',
    },
    {
      icon: Shield,
      title: 'Privacy-First Design',
      description: 'MediaPipe pose estimation creates anonymous skeleton views, ensuring no identifiable information is ever stored or processed.',
      color: 'text-healthcare-400',
    },
    {
      icon: Zap,
      title: 'Real-time Processing',
      description: 'Sub-second latency ensures immediate response to potential falls, meeting healthcare industry requirements for rapid intervention.',
      color: 'text-warning-400',
    },
    {
      icon: Building2,
      title: 'Healthcare Optimized',
      description: 'Specifically designed for healthcare environments with HIPAA compliance, alert fatigue management, and clinical workflow integration.',
      color: 'text-danger-400',
    },
  ]

  const benefits = [
    {
      icon: Shield,
      title: 'Enhanced Safety',
      description: 'Immediate fall detection with 94.2% accuracy prevents missed incidents and reduces response times.',
    },
    {
      icon: Lock,
      title: 'Privacy Protection',
      description: 'No facial recognition or identifiable data - only anonymous movement patterns are analyzed.',
    },
    {
      icon: Activity,
      title: 'Real-time Monitoring',
      description: 'Continuous 24/7 monitoring with sub-second response times for immediate intervention.',
    },
    {
      icon: Target,
      title: 'Healthcare Compliance',
      description: 'Full HIPAA compliance with healthcare-specific optimizations and clinical workflow integration.',
    },
  ]

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.1,
      },
    },
  }

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.6,
        ease: 'easeOut',
      },
    },
  }

  return (
    <section id="about" className="py-20 bg-dark-800/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="section-header">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            About the Technology
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            viewport={{ once: true }}
          >
            Cutting-edge AI for healthcare safety and privacy
          </motion.p>
        </div>

        {/* Technology Features */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16"
        >
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              variants={itemVariants}
              className="card p-6 text-center card-hover"
            >
              <div className={`w-16 h-16 mx-auto mb-4 rounded-full bg-dark-700/50 flex items-center justify-center ${feature.color}`}>
                <feature.icon className="w-8 h-8" />
              </div>
              <h3 className="text-lg font-semibold mb-3">{feature.title}</h3>
              <p className="text-dark-400 text-sm leading-relaxed">{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>

        {/* Benefits Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <div className="text-center mb-12">
            <h3 className="text-3xl font-bold gradient-text mb-4">Why Choose AI FallGuard?</h3>
            <p className="text-lg text-dark-400 max-w-3xl mx-auto">
              Our privacy-preserving fall detection system offers unmatched accuracy, 
              real-time performance, and healthcare compliance for modern care facilities.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {benefits.map((benefit, index) => (
              <motion.div
                key={benefit.title}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="card p-6 text-center"
              >
                <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-primary-500/20 flex items-center justify-center text-primary-400">
                  <benefit.icon className="w-6 h-6" />
                </div>
                <h4 className="font-semibold mb-2">{benefit.title}</h4>
                <p className="text-sm text-dark-400">{benefit.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Technology Stack */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="card p-8"
        >
          <div className="text-center mb-8">
            <h3 className="text-2xl font-bold gradient-text mb-4">Advanced Technology Stack</h3>
            <p className="text-dark-400">
              Built with cutting-edge technologies for optimal performance and reliability
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <h4 className="font-semibold mb-4 text-primary-400">Computer Vision</h4>
              <div className="space-y-2 text-sm text-dark-400">
                <div>• MediaPipe Pose Estimation</div>
                <div>• OpenCV Processing</div>
                <div>• Real-time Video Analysis</div>
                <div>• Privacy-Safe Keypoints</div>
              </div>
            </div>

            <div className="text-center">
              <h4 className="font-semibold mb-4 text-healthcare-400">Machine Learning</h4>
              <div className="space-y-2 text-sm text-dark-400">
                <div>• LSTM Neural Networks</div>
                <div>• CNN Architecture</div>
                <div>• TensorFlow Framework</div>
                <div>• Healthcare-Specific Training</div>
              </div>
            </div>

            <div className="text-center">
              <h4 className="font-semibold mb-4 text-warning-400">Real-time Processing</h4>
              <div className="space-y-2 text-sm text-dark-400">
                <div>• Sub-second Latency</div>
                <div>• Live Video Streaming</div>
                <div>• Immediate Alert Generation</div>
                <div>• Performance Optimization</div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center mt-16"
        >
          <div className="card p-8 max-w-2xl mx-auto">
            <h3 className="text-2xl font-bold gradient-text mb-4">
              Ready to Transform Healthcare Safety?
            </h3>
            <p className="text-dark-400 mb-6">
              Experience the future of fall detection with our privacy-preserving AI technology. 
              Join healthcare facilities worldwide in improving patient safety and care quality.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="btn btn-primary"
              >
                Request Demo
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="btn btn-secondary"
              >
                Learn More
              </motion.button>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
} 