export interface NewsItem {
  title: string;
  content: string;
  cash_formatted: string;
  date: string;
  total_words: number;
  brutal_words: number;
  brutal_prop: number;
  hack_type: string;
  hacker_group: string;
  confidence: number;
  normalized_cash: number;
  // SARS route specific fields
  from?: string;
  to?: string;
  hash?: string;
  timestamp?: string;
  value?: string | number;
  type?: string;
  token?: string;
  data?: any; // For additional data that might come from the API
}

export interface Source {
  name: string;
  description: string;
  url: string;
}

export interface WalletContextType {
  isConnected: boolean;
  account: string | null;
  connectWallet: () => Promise<void>;
  signMessage: () => Promise<string | null>;
  disconnectWallet: () => void;
}

export interface NewsContextType {
  selectedNews: NewsItem | null;
  setSelectedNews: (news: NewsItem | null) => void;
}

export interface SourceContextType {
  selectedSource: Source | null;
  setSelectedSource: (source: Source | null) => void;
} 