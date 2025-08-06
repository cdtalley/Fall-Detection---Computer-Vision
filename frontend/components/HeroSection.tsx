'use client'

import { motion } from 'framer-motion'
import { Brain, Eye, Lock, Zap, Activity, TrendingUp } from 'lucide-react'

interface HeroSectionProps {
  onStartDemo: () => void
}

export default function HeroSection({ onStartDemo }: HeroSectionProps) {
  const stats = [
    { label: 'Model Accuracy', value: '94.2%', icon: Brain },
    { label: 'Privacy Preserved', value: '100%', icon: Lock },
    { label: 'Real-time Processing', value: '<50ms', icon: Zap },
    { label: 'Detection Rate', value: '96.8%', icon: Eye },
  ]

  return (
    <section id="home" className="min-h-screen flex items-center justify-center relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 bg-gradient-to-br from-dark-900 via-dark-800 to-dark-900" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(59,130,246,0.1),transparent_50%)]" />
      
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        {/* Main Heading */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="mb-8"
        >
          <h1 className="text-5xl md:text-7xl font-bold mb-6">
            <span className="gradient-text">Privacy-Preserving</span>
            <br />
            <span className="text-white">Fall Detection</span>
          </h1>
          <p className="text-xl md:text-2xl text-dark-300 max-w-4xl mx-auto leading-relaxed">
            Advanced computer vision system using pose estimation for healthcare applications. 
            Demonstrates expertise in building production-ready AI systems with privacy-first design.
          </p>
        </motion.div>

        {/* Key Features */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-12"
        >
          <div className="flex flex-wrap justify-center gap-6 mb-8">
            <div className="flex items-center space-x-2 px-4 py-2 rounded-full bg-primary-500/10 border border-primary-500/20">
              <Brain className="w-5 h-5 text-primary-400" />
              <span className="text-primary-300 font-medium">Computer Vision</span>
            </div>
            <div className="flex items-center space-x-2 px-4 py-2 rounded-full bg-healthcare-500/10 border border-healthcare-500/20">
              <Lock className="w-5 h-5 text-healthcare-400" />
              <span className="text-healthcare-300 font-medium">Privacy-First</span>
            </div>
            <div className="flex items-center space-x-2 px-4 py-2 rounded-full bg-success-500/10 border border-success-500/20">
              <Zap className="w-5 h-5 text-success-400" />
              <span className="text-success-300 font-medium">Real-time</span>
            </div>
            <div className="flex items-center space-x-2 px-4 py-2 rounded-full bg-warning-500/10 border border-warning-500/20">
              <Activity className="w-5 h-5 text-warning-400" />
              <span className="text-warning-300 font-medium">Production-Ready</span>
            </div>
          </div>
        </motion.div>

        {/* Statistics */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-12"
        >
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.6 + index * 0.1 }}
              className="text-center"
            >
              <div className="flex justify-center mb-3">
                <div className="p-3 rounded-full bg-dark-700/50 border border-dark-600">
                  <stat.icon className="w-6 h-6 text-primary-400" />
                </div>
              </div>
              <div className="text-2xl md:text-3xl font-bold gradient-text mb-1">
                {stat.value}
              </div>
              <div className="text-sm text-dark-400">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="flex flex-col sm:flex-row gap-4 justify-center"
        >
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onStartDemo}
            className="px-8 py-4 bg-gradient-to-r from-primary-500 to-primary-600 text-white font-semibold rounded-lg shadow-lg hover:shadow-xl transition-all duration-200 flex items-center justify-center space-x-2"
          >
            <TrendingUp className="w-5 h-5" />
            <span>View Model Analysis</span>
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="px-8 py-4 bg-dark-700/50 border border-dark-600 text-white font-semibold rounded-lg hover:bg-dark-700 transition-all duration-200"
          >
            View Source Code
          </motion.button>
        </motion.div>

        {/* Scroll Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 1 }}
          className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-6 h-10 border-2 border-dark-400 rounded-full flex justify-center"
          >
            <motion.div
              animate={{ y: [0, 12, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-1 h-3 bg-dark-400 rounded-full mt-2"
            />
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
} 