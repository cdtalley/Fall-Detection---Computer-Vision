'use client'

import { motion } from 'framer-motion'
import { Shield, Mail, Phone, MapPin, Github, Linkedin, Twitter } from 'lucide-react'

export default function Footer() {
  const footerSections = [
    {
      title: 'AI FallGuard',
      description: 'Revolutionary privacy-preserving fall detection technology for healthcare facilities.',
      icon: Shield,
    },
    {
      title: 'Technology',
      links: [
        { name: 'Computer Vision', href: '#' },
        { name: 'Machine Learning', href: '#' },
        { name: 'Real-time Processing', href: '#' },
        { name: 'Privacy Protection', href: '#' },
      ],
    },
    {
      title: 'Healthcare',
      links: [
        { name: 'Fall Prevention', href: '#' },
        { name: 'Patient Safety', href: '#' },
        { name: 'HIPAA Compliance', href: '#' },
        { name: 'Clinical Integration', href: '#' },
      ],
    },
    {
      title: 'Contact',
      links: [
        { name: 'Demo Request', href: '#' },
        { name: 'Technical Support', href: '#' },
        { name: 'Healthcare Solutions', href: '#' },
        { name: 'Partnership', href: '#' },
      ],
    },
  ]

  const socialLinks = [
    { name: 'GitHub', icon: Github, href: '#' },
    { name: 'LinkedIn', icon: Linkedin, href: '#' },
    { name: 'Twitter', icon: Twitter, href: '#' },
  ]

  const contactInfo = [
    { icon: Mail, text: 'contact@aifallguard.com' },
    { icon: Phone, text: '+1 (555) 123-4567' },
    { icon: MapPin, text: 'San Francisco, CA' },
  ]

  return (
    <footer className="bg-dark-900 border-t border-dark-700">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-8">
          {footerSections.map((section, index) => (
            <motion.div
              key={section.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
            >
              {index === 0 ? (
                // Company section
                <div>
                  <div className="flex items-center space-x-2 mb-4">
                    {section.icon && <section.icon className="w-8 h-8 text-primary-500" />}
                    <span className="text-xl font-bold gradient-text">AI FallGuard</span>
                  </div>
                  <p className="text-dark-400 text-sm leading-relaxed mb-4">
                    {section.description}
                  </p>
                  
                  {/* Contact Info */}
                  <div className="space-y-2">
                    {contactInfo.map((contact, idx) => (
                      <div key={idx} className="flex items-center space-x-2 text-sm text-dark-400">
                        <contact.icon className="w-4 h-4" />
                        <span>{contact.text}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                // Links sections
                <div>
                  <h3 className="text-lg font-semibold mb-4">{section.title}</h3>
                  <ul className="space-y-2">
                    {section.links?.map((link, idx) => (
                      <li key={idx}>
                        <motion.a
                          href={link.href}
                          whileHover={{ x: 5 }}
                          className="text-sm text-dark-400 hover:text-white transition-colors duration-200"
                        >
                          {link.name}
                        </motion.a>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </motion.div>
          ))}
        </div>

        {/* Social Links */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="flex justify-center space-x-6 mb-8"
        >
          {socialLinks.map((social) => (
            <motion.a
              key={social.name}
              href={social.href}
              whileHover={{ scale: 1.1, y: -2 }}
              whileTap={{ scale: 0.95 }}
              className="w-10 h-10 bg-dark-700 rounded-full flex items-center justify-center text-dark-400 hover:text-primary-400 hover:bg-dark-600 transition-all duration-200"
            >
              <social.icon className="w-5 h-5" />
            </motion.a>
          ))}
        </motion.div>

        {/* Bottom Bar */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="border-t border-dark-700 pt-8"
        >
          <div className="flex flex-col md:flex-row justify-between items-center">
            <p className="text-sm text-dark-400 mb-4 md:mb-0">
              &copy; 2024 AI FallGuard. All rights reserved. Built for healthcare innovation.
            </p>
            <div className="flex space-x-6 text-sm text-dark-400">
              <a href="#" className="hover:text-white transition-colors duration-200">
                Privacy Policy
              </a>
              <a href="#" className="hover:text-white transition-colors duration-200">
                Terms of Service
              </a>
              <a href="#" className="hover:text-white transition-colors duration-200">
                Cookie Policy
              </a>
            </div>
          </div>
        </motion.div>
      </div>
    </footer>
  )
} 