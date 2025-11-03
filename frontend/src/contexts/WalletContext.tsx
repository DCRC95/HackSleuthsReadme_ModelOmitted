import React, { createContext, useContext, useState, useEffect } from 'react';
import { ethers } from 'ethers';
import { WalletContextType } from '../types';

declare global {
  interface Window {
    ethereum?: any;
  }
}

const WalletContext = createContext<WalletContextType | undefined>(undefined);

export const WalletProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [account, setAccount] = useState<string | null>(null);

  useEffect(() => {
    checkConnection();
    const ethereum = window.ethereum;
    if (ethereum?.on) {
      ethereum.on('accountsChanged', handleAccountsChanged);
      ethereum.on('chainChanged', () => window.location.reload());
    }
    return () => {
      if (ethereum?.removeListener) {
        ethereum.removeListener('accountsChanged', handleAccountsChanged);
      }
    };
  }, []);

  const checkConnection = async () => {
    try {
      const ethereum = window.ethereum;
      if (!ethereum) return;

      const provider = new ethers.BrowserProvider(ethereum);
      const accounts = await provider.listAccounts();
      if (accounts.length > 0) {
        setAccount(accounts[0].address);
        setIsConnected(true);
      }
    } catch (error) {
      console.error('Error checking connection:', error);
    }
  };

  const handleAccountsChanged = (accounts: string[]) => {
    if (accounts.length > 0) {
      setAccount(accounts[0]);
      setIsConnected(true);
    } else {
      setAccount(null);
      setIsConnected(false);
    }
  };

  const connectWallet = async () => {
    try {
      const ethereum = window.ethereum;
      if (!ethereum) {
        alert('Please install MetaMask to use this feature!');
        return;
      }

      const accounts = await ethereum.request({
        method: 'eth_requestAccounts'
      });
      
      if (accounts && accounts.length > 0) {
        setAccount(accounts[0]);
        setIsConnected(true);
      }
    } catch (error) {
      console.error('Error connecting wallet:', error);
      throw error;
    }
  };

  const signMessage = async () => {
    try {
      const ethereum = window.ethereum;
      if (!ethereum || !account) return null;

      const provider = new ethers.BrowserProvider(ethereum);
      const signer = await provider.getSigner();
      const message = `Welcome to CryptoPanic!\nPlease sign this message to verify your identity.\n\nTimestamp: ${Date.now()}`;
      const signature = await signer.signMessage(message);
      return signature;
    } catch (error) {
      console.error('Error signing message:', error);
      return null;
    }
  };

  const disconnectWallet = () => {
    setAccount(null);
    setIsConnected(false);
  };

  return (
    <WalletContext.Provider value={{
      isConnected,
      account,
      connectWallet,
      signMessage,
      disconnectWallet
    }}>
      {children}
    </WalletContext.Provider>
  );
};

export const useWallet = () => {
  const context = useContext(WalletContext);
  if (context === undefined) {
    throw new Error('useWallet must be used within a WalletProvider');
  }
  return context;
}; 