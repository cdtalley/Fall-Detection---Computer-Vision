import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Toaster } from 'react-hot-toast'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'AI FallGuard - Privacy-Preserving Fall Detection System',
  description: 'Revolutionary AI-powered fall detection system with 94.2% accuracy and sub-second response times. HIPAA-compliant, privacy-preserving computer vision technology for healthcare.',
  keywords: 'fall detection, AI, computer vision, healthcare, privacy, MediaPipe, pose estimation',
  authors: [{ name: 'AI FallGuard Team' }],
  viewport: 'width=device-width, initial-scale=1',
  robots: 'index, follow',
  openGraph: {
    title: 'AI FallGuard - Privacy-Preserving Fall Detection System',
    description: 'Revolutionary AI-powered fall detection system with 94.2% accuracy and sub-second response times.',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'AI FallGuard - Privacy-Preserving Fall Detection System',
    description: 'Revolutionary AI-powered fall detection system with 94.2% accuracy and sub-second response times.',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${inter.className} antialiased bg-dark-900 text-white overflow-x-hidden`}>
        {children}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#1e293b',
              color: '#f8fafc',
              border: '1px solid #334155',
            },
            success: {
              iconTheme: {
                primary: '#22c55e',
                secondary: '#f8fafc',
              },
            },
            error: {
              iconTheme: {
                primary: '#ef4444',
                secondary: '#f8fafc',
              },
            },
          }}
        />
      </body>
    </html>
  )
} 