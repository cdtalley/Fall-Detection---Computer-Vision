'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Shield, Menu, X, Wifi, WifiOff } from 'lucide-react'

interface NavigationProps {
  activeSection: string
  onSectionChange: (section: string) => void
}

export default function Navigation({ activeSection, onSectionChange }: NavigationProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [isOnline, setIsOnline] = useState(true)

  useEffect(() => {
    // Simulate system status
    const interval = setInterval(() => {
      setIsOnline(Math.random() > 0.1) // 90% uptime simulation
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const navItems = [
    { id: 'home', label: 'Home', href: '#home' },
    { id: 'demo', label: 'Live Demo', href: '#demo' },
    { id: 'analytics', label: 'Analytics', href: '#analytics' },
    { id: 'about', label: 'About', href: '#about' },
  ]

  const handleNavClick = (sectionId: string) => {
    onSectionChange(sectionId)
    setIsOpen(false)
  }

  return (
    <motion.nav
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="fixed top-0 left-0 right-0 z-50 bg-dark-900/80 backdrop-blur-md border-b border-dark-700"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="flex items-center space-x-2 cursor-pointer"
            onClick={() => handleNavClick('home')}
          >
            <Shield className="w-8 h-8 text-primary-500" />
            <span className="text-xl font-bold gradient-text">AI FallGuard</span>
          </motion.div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            {navItems.map((item) => (
              <motion.button
                key={item.id}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => handleNavClick(item.id)}
                className={`relative px-3 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                  activeSection === item.id
                    ? 'text-primary-400 bg-primary-500/10'
                    : 'text-dark-300 hover:text-white hover:bg-dark-700/50'
                }`}
              >
                {item.label}
                {activeSection === item.id && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute inset-0 bg-primary-500/10 rounded-md border border-primary-500/20"
                    initial={false}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                  />
                )}
              </motion.button>
            ))}
          </div>

          {/* System Status */}
          <div className="hidden md:flex items-center space-x-4">
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center space-x-2 px-3 py-1 rounded-full bg-dark-700/50 border border-dark-600"
            >
              {isOnline ? (
                <Wifi className="w-4 h-4 text-healthcare-500" />
              ) : (
                <WifiOff className="w-4 h-4 text-danger-500" />
              )}
              <span className="text-xs font-medium">
                {isOnline ? 'System Online' : 'System Offline'}
              </span>
            </motion.div>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <motion.button
              whileTap={{ scale: 0.95 }}
              onClick={() => setIsOpen(!isOpen)}
              className="p-2 rounded-md text-dark-300 hover:text-white hover:bg-dark-700/50"
            >
              {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </motion.button>
          </div>
        </div>

        {/* Mobile Navigation */}
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden border-t border-dark-700"
            >
              <div className="px-2 pt-2 pb-3 space-y-1">
                {navItems.map((item) => (
                  <motion.button
                    key={item.id}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => handleNavClick(item.id)}
                    className={`block w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200 ${
                      activeSection === item.id
                        ? 'text-primary-400 bg-primary-500/10'
                        : 'text-dark-300 hover:text-white hover:bg-dark-700/50'
                    }`}
                  >
                    {item.label}
                  </motion.button>
                ))}
                
                {/* Mobile System Status */}
                <div className="px-3 py-2">
                  <div className="flex items-center space-x-2">
                    {isOnline ? (
                      <Wifi className="w-4 h-4 text-healthcare-500" />
                    ) : (
                      <WifiOff className="w-4 h-4 text-danger-500" />
                    )}
                    <span className="text-xs font-medium text-dark-400">
                      {isOnline ? 'System Online' : 'System Offline'}
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.nav>
  )
} 