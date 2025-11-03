# HackSleuth

A modern cryptocurrency analysis aggregator built with React, TypeScript, and Chakra UI.

## Features

- Real-time cryptocurrency news aggregation
- Web3 wallet integration for user interactions
- News filtering by currency
- Global search functionality
- Reaction system (likes, dislikes, important)
- Social sharing capabilities
- Responsive and modern UI
- Dark theme

## Tech Stack

- React 18
- TypeScript
- Chakra UI
- React Router
- Ethers.js
- React Icons

## Prerequisites

- Node.js 16+ and npm
- MetaMask or another Web3 wallet

## Getting Started

1. Clone the repository:
```bash
git clone <repository-url>
cd cryptopanic-v2
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The application will be available at `http://localhost:3000`.

## Project Structure

```
src/
├── components/         # React components
├── contexts/          # Context providers
├── layouts/           # Layout components
├── types/            # TypeScript types
├── App.tsx           # Main App component
├── index.tsx         # Entry point
└── theme.ts          # Chakra UI theme configuration
```

## Development

### Available Scripts

- `npm start` - Runs the app in development mode
- `npm build` - Builds the app for production
- `npm test` - Runs the test suite
- `npm eject` - Ejects from Create React App

### Code Style

The project uses TypeScript for type safety and follows React best practices. Components are functional and use hooks for state management.

## Features to Implement

### Phase 1 (Core)
- [x] News feed with basic filtering
- [x] Source information display
- [x] Basic wallet connection
- [x] News detail view

### Phase 2 (Engagement)
- [ ] Reaction system backend integration
- [x] Share functionality
- [ ] Search implementation
- [ ] Trending section

### Phase 3 (Personalization)
- [ ] Watchlist functionality
- [ ] User settings
- [ ] Advanced filtering
- [ ] Notifications

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. 