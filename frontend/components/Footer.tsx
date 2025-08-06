'use client'

import { motion } from 'framer-motion'
import { Brain, Github, Linkedin, Mail, Code, Database, BookOpen } from 'lucide-react'

export default function Footer() {
  const sections = [
    {
      title: 'Project',
      items: [
        { label: 'GitHub Repository', href: '#', icon: Github },
        { label: 'Documentation', href: '#', icon: BookOpen },
        { label: 'Dataset Download', href: '#', icon: Database },
        { label: 'API Reference', href: '#', icon: Code },
      ]
    },
    {
      title: 'Technology',
      items: [
        { label: 'Computer Vision', href: '#', icon: Brain },
        { label: 'Machine Learning', href: '#', icon: Brain },
        { label: 'Privacy & Security', href: '#', icon: Brain },
        { label: 'Real-time Processing', href: '#', icon: Brain },
      ]
    },
    {
      title: 'Connect',
      items: [
        { label: 'LinkedIn', href: '#', icon: Linkedin },
        { label: 'Email', href: '#', icon: Mail },
        { label: 'GitHub Profile', href: '#', icon: Github },
        { label: 'Portfolio', href: '#', icon: Code },
      ]
    }
  ]

  return (
    <footer className="bg-dark-900 border-t border-dark-700">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Main Footer Content */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-8">
          {/* Project Info */}
          <div className="lg:col-span-1">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="flex items-center space-x-2 mb-4"
            >
              <Brain className="w-8 h-8 text-primary-500" />
              <span className="text-xl font-bold gradient-text">Privacy-Preserving Fall Detection</span>
            </motion.div>
            <p className="text-dark-300 mb-4 leading-relaxed">
              Advanced computer vision system demonstrating expertise in AI/ML, 
              privacy-preserving design, and production-ready architecture for healthcare applications.
            </p>
            <div className="flex space-x-4">
              <motion.a
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                href="#"
                className="p-2 bg-dark-700 rounded-lg hover:bg-primary-500/20 transition-colors duration-200"
              >
                <Github className="w-5 h-5 text-dark-300 hover:text-primary-400" />
              </motion.a>
              <motion.a
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                href="#"
                className="p-2 bg-dark-700 rounded-lg hover:bg-primary-500/20 transition-colors duration-200"
              >
                <Linkedin className="w-5 h-5 text-dark-300 hover:text-primary-400" />
              </motion.a>
              <motion.a
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                href="#"
                className="p-2 bg-dark-700 rounded-lg hover:bg-primary-500/20 transition-colors duration-200"
              >
                <Mail className="w-5 h-5 text-dark-300 hover:text-primary-400" />
              </motion.a>
            </div>
          </div>

          {/* Footer Sections */}
          {sections.map((section, sectionIndex) => (
            <motion.div
              key={section.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: sectionIndex * 0.1 }}
              viewport={{ once: true }}
            >
              <h3 className="text-lg font-semibold text-white mb-4">{section.title}</h3>
              <ul className="space-y-3">
                {section.items.map((item, itemIndex) => (
                  <motion.li
                    key={item.label}
                    initial={{ opacity: 0, x: -10 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.4, delay: (sectionIndex * 0.1) + (itemIndex * 0.05) }}
                    viewport={{ once: true }}
                  >
                    <a
                      href={item.href}
                      className="flex items-center space-x-2 text-dark-300 hover:text-primary-400 transition-colors duration-200"
                    >
                      {item.icon && <item.icon className="w-4 h-4" />}
                      <span>{item.label}</span>
                    </a>
                  </motion.li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>

        {/* Bottom Bar */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="pt-8 border-t border-dark-700"
        >
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="text-dark-400 text-sm mb-4 md:mb-0">
              © 2024 Privacy-Preserving Fall Detection. Open source project for research and educational purposes.
            </div>
            <div className="flex space-x-6 text-sm">
              <a href="#" className="text-dark-400 hover:text-primary-400 transition-colors duration-200">
                Privacy Policy
              </a>
              <a href="#" className="text-dark-400 hover:text-primary-400 transition-colors duration-200">
                Terms of Use
              </a>
              <a href="#" className="text-dark-400 hover:text-primary-400 transition-colors duration-200">
                License
              </a>
            </div>
          </div>
        </motion.div>
      </div>
    </footer>
  )
} 