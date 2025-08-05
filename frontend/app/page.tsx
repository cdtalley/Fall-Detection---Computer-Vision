'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Shield, Activity, Zap, Eye, Brain, UserCheck, Lock, Building2 } from 'lucide-react'
import toast from 'react-hot-toast'

import Navigation from '@/components/Navigation'
import HeroSection from '@/components/HeroSection'
import ModelPerformance from '@/components/LiveDemo'
import Analytics from '@/components/Analytics'
import DatasetInfo from '@/components/DatasetInfo'
import AboutSection from '@/components/AboutSection'
import Footer from '@/components/Footer'
import ParticleBackground from '@/components/ParticleBackground'

export default function Home() {
  const [isLoading, setIsLoading] = useState(true)
  const [activeSection, setActiveSection] = useState('home')

  useEffect(() => {
    // Simulate loading
    const timer = setTimeout(() => {
      setIsLoading(false)
    }, 1000)

    return () => clearTimeout(timer)
  }, [])

  useEffect(() => {
    const handleScroll = () => {
      const sections = ['home', 'model-performance', 'analytics', 'datasets', 'about']
      const scrollPosition = window.scrollY + 100

      for (const section of sections) {
        const element = document.getElementById(section)
        if (element) {
          const { offsetTop, offsetHeight } = element
          if (scrollPosition >= offsetTop && scrollPosition < offsetTop + offsetHeight) {
            setActiveSection(section)
            break
          }
        }
      }
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' })
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-dark-900 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center"
        >
          <div className="relative">
            <Shield className="w-16 h-16 text-primary-500 mx-auto mb-4 animate-pulse" />
            <div className="text-2xl font-bold gradient-text">AI FallGuard</div>
            <div className="text-dark-400 mt-2">Loading revolutionary fall detection system...</div>
          </div>
        </motion.div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-dark-900">
      <ParticleBackground />
      
      <Navigation activeSection={activeSection} onSectionChange={scrollToSection} />
      
      <main>
        <HeroSection onStartDemo={() => scrollToSection('model-performance')} />
        
        <ModelPerformance />
        
        <Analytics />
        
        <section id="datasets" className="py-20">
          <DatasetInfo />
        </section>
        
        <AboutSection />
      </main>
      
      <Footer />
    </div>
  )
} 