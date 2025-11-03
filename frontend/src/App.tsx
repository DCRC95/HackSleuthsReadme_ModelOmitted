import React from 'react';
import { ChakraProvider } from '@chakra-ui/react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import theme from './theme';
import MainLayout from './layouts/MainLayout';
import MainContent from './components/MainContent';
import { WalletProvider } from './contexts/WalletContext';
import { SourceProvider } from './contexts/SourceContext';
import { NewsProvider } from './contexts/NewsContext';

const App: React.FC = () => {
  return (
    <ChakraProvider theme={theme}>
      <WalletProvider>
        <SourceProvider>
          <NewsProvider>
            <Router>
              <Routes>
                <Route path="/" element={<MainLayout />}>
                  <Route index element={<MainContent />} />
                  <Route path="sars" element={<MainContent />} />
                  <Route path="sars/:title" element={<MainContent />} />
                  <Route path="watchlist" element={<MainContent />} />
                  <Route path="settings" element={<MainContent />} />
                </Route>
              </Routes>
            </Router>
          </NewsProvider>
        </SourceProvider>
      </WalletProvider>
    </ChakraProvider>
  );
};

export default App; 