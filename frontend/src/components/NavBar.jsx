import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { trackEvent } from '../lib/posthog';
import './NavBar.css';

const NavBar = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const location = useLocation();

  const navLinks = [
    { to: '/home', text: 'Home' },
    { to: '/dashboard', text: 'Dashboard' },
    { to: '/model',   text: 'Predict Rent' },
    { to: '/compare', text: 'Compare Models' },
    { to: '/about',   text: 'About' }
  ];

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
    trackEvent('mobile_menu_toggled', { isOpen: !isMobileMenuOpen });
  };

  const handleNavClick = (link) => {
    setIsMobileMenuOpen(false);
    trackEvent('navigation_clicked', {
      destination: link.to,
      label: link.text,
      source: isMobileMenuOpen ? 'mobile_menu' : 'desktop_nav',
    });
  };

  return (
    <header>
      <nav className="navbar">
        <div className="container">
          <Link className="navbar-brand" to="/">🗼 Tokyo Rent</Link>
          <div className="navbar-links">
            <ul className="navbar-nav">
              {navLinks.map((link) => (
                <li key={link.to} className="nav-item">
                  <Link
                    className={`nav-link${location.pathname === link.to ? ' active' : ''}`}
                    to={link.to}
                    onClick={() => handleNavClick(link)}
                  >
                    {link.text}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
          <button
            onClick={toggleMobileMenu}
            className="mobile-menu-button"
            aria-label="Toggle mobile menu"
            aria-expanded={isMobileMenuOpen}
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16m-7 6h7" />
            </svg>
          </button>
        </div>
        {isMobileMenuOpen && (
          <div className="mobile-menu">
            <ul className="navbar-nav">
              {navLinks.map((link) => (
                <li key={link.to} className="nav-item">
                  <Link
                    className={`nav-link${location.pathname === link.to ? ' active' : ''}`}
                    to={link.to}
                    onClick={() => handleNavClick(link)}
                  >
                    {link.text}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        )}
      </nav>
    </header>
  );
};

export default NavBar;
