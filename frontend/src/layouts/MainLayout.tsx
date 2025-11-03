import React from 'react';
import { Box } from '@chakra-ui/react';
import { Outlet, useLocation } from 'react-router-dom';
import Header from '../components/Header';
import LeftSidebar from '../components/LeftSidebar';
import RightSidebar from '../components/RightSidebar';
import { useNews } from '../contexts/NewsContext';

const MainLayout: React.FC = () => {
  const location = useLocation();
  const { selectedNews } = useNews();
  const isSarsRoute = location.pathname.startsWith('/sars');

  return (
    <Box minH="100vh" bg="brand.background">
      <Header />
      <LeftSidebar />
      <Box>
        <Outlet />
      </Box>
      {!isSarsRoute && selectedNews && <RightSidebar />}
    </Box>
  );
};

export default MainLayout; 