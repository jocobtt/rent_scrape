import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { ToastProvider } from './context/ToastContext'
import { PostHogProvider } from './context/PostHogContext'
import NavBar from './components/NavBar'
import Footer from './components/Footer'
import Home from './pages/Home'
import Dashboard from './pages/Dashboard'
import Model from './pages/Model'
import Compare from './pages/Compare'
import About from './pages/About'
import NotFound from './pages/NotFound'

function App() {
  return (
    <ToastProvider>
      <Router>
        <PostHogProvider>
          <div className="app-container">
            <NavBar />
            <main className="main-content">
              <Routes>
                <Route path="/" element={<Navigate to="/home" replace />} />
                <Route path="/home" element={<Home />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/model" element={<Model />} />
                <Route path="/compare" element={<Compare />} />
                <Route path="/about" element={<About />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </main>
            <Footer />
          </div>
        </PostHogProvider>
      </Router>
    </ToastProvider>
  )
}

export default App
