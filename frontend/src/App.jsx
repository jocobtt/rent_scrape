import React, { Suspense, lazy } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { ToastProvider } from './context/ToastContext'
import { PostHogProvider } from './context/PostHogContext'
import NavBar from './components/NavBar'
import Footer from './components/Footer'
import Home from './pages/Home'

// Home is the landing page; the rest load on demand (Dashboard pulls in all of D3)
const Dashboard = lazy(() => import('./pages/Dashboard'))
const Model = lazy(() => import('./pages/Model'))
const Compare = lazy(() => import('./pages/Compare'))
const About = lazy(() => import('./pages/About'))
const NotFound = lazy(() => import('./pages/NotFound'))

function App() {
  return (
    <ToastProvider>
      <Router>
        <PostHogProvider>
          <div className="app-container">
            <NavBar />
            <main className="main-content">
              <Suspense fallback={<div className="route-loading" role="status">Loading…</div>}>
              <Routes>
                <Route path="/" element={<Navigate to="/home" replace />} />
                <Route path="/home" element={<Home />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/model" element={<Model />} />
                <Route path="/compare" element={<Compare />} />
                <Route path="/about" element={<About />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
              </Suspense>
            </main>
            <Footer />
          </div>
        </PostHogProvider>
      </Router>
    </ToastProvider>
  )
}

export default App
