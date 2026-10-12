module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:react/jsx-runtime',
    'plugin:react-hooks/recommended',
  ],
  ignorePatterns: ['dist', 'node_modules'],
  parserOptions: { ecmaVersion: 'latest', sourceType: 'module' },
  settings: { react: { version: '18.3' } },
  plugins: ['react-refresh'],
  overrides: [
    {
      files: ['*.config.js', '*.config.cjs'],
      env: { node: true },
    },
  ],
  rules: {
    'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
    // React 17+ JSX transform doesn't require React in scope
    'no-unused-vars': ['warn', { varsIgnorePattern: '^(_|React$)', argsIgnorePattern: '^_' }],
    // PropTypes not used (TypeScript migration planned)
    'react/prop-types': 'off',
  },
}
