'use client'

import { motion } from 'framer-motion'
import { Play, Eye, Shield, Zap, Activity } from 'lucide-react'

interface HeroSectionProps {
  onStartDemo: () => void
}

export default function HeroSection({ onStartDemo }: HeroSectionProps) {
  const stats = [
    { value: '94.2%', label: 'Sensitivity', icon: Shield },
    { value: '<500ms', label: 'Response Time', icon: Zap },
    { value: '100%', label: 'Privacy Safe', icon: Activity },
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
    <section id="home" className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-dark-900 via-dark-800 to-dark-900">
        <div className="absolute inset-0 bg-mesh-gradient opacity-20"></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Content */}
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="text-center lg:text-left"
          >
            <motion.h1
              variants={itemVariants}
              className="text-5xl lg:text-7xl font-bold leading-tight mb-6"
            >
              <span className="gradient-text">AI-Powered</span>
              <br />
              <span className="text-white">Fall Detection</span>
              <br />
              <span className="text-4xl lg:text-5xl text-dark-300">System</span>
            </motion.h1>

            <motion.p
              variants={itemVariants}
              className="text-xl lg:text-2xl text-dark-300 mb-8 max-w-2xl mx-auto lg:mx-0"
            >
              Revolutionary privacy-preserving computer vision technology for healthcare.
              Real-time monitoring with 94.2% accuracy and sub-second response times.
            </motion.p>

            {/* Stats */}
            <motion.div
              variants={itemVariants}
              className="grid grid-cols-3 gap-6 mb-8"
            >
              {stats.map((stat, index) => (
                <motion.div
                  key={stat.label}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.5 + index * 0.1, type: 'spring', stiffness: 200 }}
                  className="text-center"
                >
                  <div className="flex items-center justify-center mb-2">
                    <stat.icon className="w-6 h-6 text-primary-400 mr-2" />
                    <div className="text-2xl lg:text-3xl font-bold gradient-text">
                      {stat.value}
                    </div>
                  </div>
                  <div className="text-sm text-dark-400 font-medium">
                    {stat.label}
                  </div>
                </motion.div>
              ))}
            </motion.div>

            {/* CTA Buttons */}
            <motion.div
              variants={itemVariants}
              className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start"
            >
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={onStartDemo}
                className="btn btn-primary text-lg px-8 py-4"
              >
                <Play className="w-5 h-5 mr-2" />
                Start Live Demo
              </motion.button>
              
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={onStartDemo}
                className="btn btn-secondary text-lg px-8 py-4"
              >
                <Eye className="w-5 h-5 mr-2" />
                View Demo
              </motion.button>
            </motion.div>
          </motion.div>

          {/* Right Visual */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="relative"
          >
            <div className="relative w-full max-w-lg mx-auto">
              {/* Skeleton Animation */}
              <div className="relative bg-dark-800/50 backdrop-blur-sm rounded-2xl p-8 border border-dark-700">
                <div className="skeleton-animation relative">
                  <div className="skeleton-person">
                    <motion.div
                      animate={{ y: [0, -5, 0] }}
                      transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                      className="head"
                    />
                    <div className="body" />
                    <div className="arms">
                      <motion.div
                        animate={{ rotate: [0, 5, 0] }}
                        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                        className="arm left"
                      />
                      <motion.div
                        animate={{ rotate: [0, -5, 0] }}
                        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                        className="arm right"
                      />
                    </div>
                    <div className="legs">
                      <motion.div
                        animate={{ rotate: [0, 3, 0] }}
                        transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
                        className="leg left"
                      />
                      <motion.div
                        animate={{ rotate: [0, -3, 0] }}
                        transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
                        className="leg right"
                      />
                    </div>
                  </div>
                  
                  {/* Detection Rings */}
                  <div className="detection-rings">
                    <div className="ring ring-1" />
                    <div className="ring ring-2" />
                    <div className="ring ring-3" />
                  </div>
                </div>

                {/* Status Indicators */}
                <div className="absolute top-4 right-4 flex space-x-2">
                  <div className="w-3 h-3 bg-healthcare-500 rounded-full animate-pulse" />
                  <div className="w-3 h-3 bg-primary-500 rounded-full animate-pulse" style={{ animationDelay: '0.5s' }} />
                  <div className="w-3 h-3 bg-warning-500 rounded-full animate-pulse" style={{ animationDelay: '1s' }} />
                </div>

                {/* Detection Scan Line */}
                <motion.div
                  animate={{ x: ['-100%', '100%'] }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                  className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent"
                />
              </div>

              {/* Floating Elements */}
              <motion.div
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                className="absolute -top-4 -left-4 w-8 h-8 bg-primary-500/20 rounded-full backdrop-blur-sm"
              />
              <motion.div
                animate={{ y: [0, 10, 0] }}
                transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
                className="absolute -bottom-4 -right-4 w-6 h-6 bg-healthcare-500/20 rounded-full backdrop-blur-sm"
              />
            </div>
          </motion.div>
        </div>
      </div>

      {/* Scroll Indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5 }}
        className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
      >
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          className="w-6 h-10 border-2 border-dark-400 rounded-full flex justify-center"
        >
          <motion.div
            animate={{ y: [0, 12, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
            className="w-1 h-3 bg-dark-400 rounded-full mt-2"
          />
        </motion.div>
      </motion.div>
    </section>
  )
} 