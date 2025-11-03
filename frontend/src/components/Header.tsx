import React from 'react';
import {
  Flex,
  Input,
  Select,
  IconButton,
  InputGroup,
  InputRightElement,
  Text,
  Link,
  HStack,
  Button,
  useToast,
  Tooltip,
  Box
} from '@chakra-ui/react';
import { SearchIcon, CopyIcon } from '@chakra-ui/icons';
import { Link as RouterLink } from 'react-router-dom';

const CONTRACT_ADDRESS = "0x76c82218f851496b86186dd824ab0b642f803f79"; // Replace with your actual contract address

const Header: React.FC = () => {
  const toast = useToast();

  const copyToClipboard = async (text: string) => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        // Fallback method
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.position = "fixed";  // Avoid scrolling to bottom
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        try {
          document.execCommand("copy");
        } catch (err) {
          console.error("Fallback: Copy failed", err);
        }
        document.body.removeChild(textArea);
      }
      toast({
        title: "Copied to clipboard",
        status: "success",
        duration: 2000,
        isClosable: true,
        position: "top"
      });
    } catch (err) {
      toast({
        title: "Failed to copy",
        status: "error",
        duration: 2000,
        isClosable: true,
        position: "top"
      });
    }
  };

  return (
    <Flex
      as="header"
      align="center"
      justify="center"
      padding="1rem"
      bg="brand.gray"
      borderBottom="1px solid"
      borderColor="whiteAlpha.200"
      height="60px"
      position="fixed"
      top={0}
      left={0}
      right={0}
      zIndex={1000}
    >
      <Box position="relative" width="100%" maxW="1200px">
        {/* Centered Title */}
        <Flex justify="center" width="100%" position="absolute" left="0">
          <Link 
            as={RouterLink} 
            to="/" 
            _hover={{ textDecoration: 'none' }}
          >
            <Text
              fontSize="2xl"
              fontWeight="bold"
              color="brand.orange"
              letterSpacing="tight"
            >
              HackSleuths
            </Text>
          </Link>
        </Flex>

        {/* Contract Address - Right */}
        <Flex justify="flex-end" pr={4}>
          <Tooltip label="Click to copy contract address" hasArrow>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => copyToClipboard(CONTRACT_ADDRESS)}
              color="whiteAlpha.700"
              _hover={{ color: 'brand.orange' }}
              leftIcon={<CopyIcon />}
            >
              <Text fontSize="xs" isTruncated maxW="200px">
                Smart Contract Address: {CONTRACT_ADDRESS}
              </Text>
            </Button>
          </Tooltip>
        </Flex>
      </Box>
    </Flex>
  );
};

export default Header; 