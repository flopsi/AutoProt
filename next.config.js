/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  transpilePackages: ['react-plotly.js', 'plotly.js'],
}

module.exports = nextConfig
