import React, { createContext, useContext, useState } from 'react';
import { SourceContextType, Source } from '../types';

const SourceContext = createContext<SourceContextType | undefined>(undefined);

export const SourceProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [selectedSource, setSelectedSource] = useState<Source | null>(null);

  return (
    <SourceContext.Provider value={{ selectedSource, setSelectedSource }}>
      {children}
    </SourceContext.Provider>
  );
};

export const useSource = () => {
  const context = useContext(SourceContext);
  if (context === undefined) {
    throw new Error('useSource must be used within a SourceProvider');
  }
  return context;
}; 