import React, { createContext, useContext, useState } from 'react';
import { NewsContextType, NewsItem } from '../types';

const NewsContext = createContext<NewsContextType | undefined>(undefined);

export const NewsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [selectedNews, setSelectedNews] = useState<NewsItem | null>(null);

  return (
    <NewsContext.Provider value={{ selectedNews, setSelectedNews }}>
      {children}
    </NewsContext.Provider>
  );
};

export const useNews = () => {
  const context = useContext(NewsContext);
  if (context === undefined) {
    throw new Error('useNews must be used within a NewsProvider');
  }
  return context;
}; 