import React from 'react';
import { VStack, Link, Icon, Text, Flex } from '@chakra-ui/react';
import { Link as RouterLink, useLocation } from 'react-router-dom';
import { FiHome, FiTrendingUp, FiStar, FiSettings } from 'react-icons/fi';

interface NavItemProps {
  icon: any;
  to: string;
  children: React.ReactNode;
}

const NavItem: React.FC<NavItemProps> = ({ icon, to, children }) => {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <Link
      as={RouterLink}
      to={to}
      _hover={{ textDecoration: 'none' }}
      width="100%"
    >
      <Flex
        align="center"
        p={3}
        mx={3}
        borderRadius="lg"
        role="group"
        cursor="pointer"
        color={isActive ? 'brand.orange' : 'whiteAlpha.700'}
        _hover={{
          bg: 'whiteAlpha.100',
          color: 'brand.orange',
        }}
      >
        <Icon
          mr={4}
          fontSize="16"
          as={icon}
        />
        <Text fontSize="sm" fontWeight={isActive ? 'bold' : 'normal'}>
          {children}
        </Text>
      </Flex>
    </Link>
  );
};

const LeftSidebar: React.FC = () => {
  return (
    <VStack
      spacing={0}
      align="stretch"
      h="calc(100vh - 60px)"
      pt={4}
      bg="brand.gray"
      position="fixed"
      top="60px"
      left={0}
      width="240px"
      borderRight="1px solid"
      borderColor="whiteAlpha.200"
    >
      <NavItem icon={FiHome} to="/">
        Hacks
      </NavItem>
      <NavItem icon={FiTrendingUp} to="/sars">
        SARS
      </NavItem>
      <NavItem icon={FiStar} to="/watchlist">
        Watchlist
      </NavItem>
      <NavItem icon={FiSettings} to="/settings">
        Settings
      </NavItem>
    </VStack>
  );
};

export default LeftSidebar; 