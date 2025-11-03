import React from 'react';
import { Box, HStack, Text, Badge, Card, CardBody, VStack, useToast, IconButton, Button, Divider } from '@chakra-ui/react';
import { useSource } from '../contexts/SourceContext';
import { useNews } from '../contexts/NewsContext';
import { NewsItem } from '../types';
import { CopyIcon } from '@chakra-ui/icons';

interface NewsCardProps {
  news: NewsItem;
  isSarsRoute: boolean;
}

const NewsCard: React.FC<NewsCardProps> = ({ news, isSarsRoute }) => {
  const { setSelectedSource } = useSource();
  const { setSelectedNews } = useNews();
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

  const handleSourceClick = () => {
    setSelectedSource({
      name: news.hacker_group,
      description: `Known hacker group with ${(news.confidence * 100).toFixed(1)}% confidence level`,
      url: '#'
    });

    copyToClipboard(JSON.stringify(news, null, 2));
  };

  const handleNewsClick = () => {
    setSelectedNews(news);
  };

  if (isSarsRoute) {
    return (
      <Card 
        bg="whiteAlpha.100" 
        borderColor="whiteAlpha.200" 
        borderWidth="1px"
        _hover={{ borderColor: 'brand.orange' }}
        transition="all 0.2s"
      >
        <CardBody py={2} px={3}>
          <VStack align="stretch" spacing={2}>
            <HStack justify="space-between">
              <Text fontSize="xs" color="whiteAlpha.600">From:</Text>
              <Text fontSize="xs" color="white" isTruncated maxW="70%">
                {news.from || 'Unknown'}
              </Text>
            </HStack>

            <HStack justify="space-between">
              <Text fontSize="xs" color="whiteAlpha.600">To:</Text>
              <Text fontSize="xs" color="white" isTruncated maxW="70%">
                {news.to || 'Unknown'}
              </Text>
            </HStack>

            <HStack justify="space-between">
              <Text fontSize="xs" color="whiteAlpha.600">Hash:</Text>
              <Text fontSize="xs" color="white" isTruncated maxW="70%">
                {news.hash || 'Unknown'}
              </Text>
            </HStack>

            <HStack justify="space-between">
              <Text fontSize="xs" color="whiteAlpha.600">Timestamp:</Text>
              <Text fontSize="xs" color="white">
                {news.timestamp || 'Unknown'}
              </Text>
            </HStack>

            <HStack justify="space-between">
              <Text fontSize="xs" color="whiteAlpha.600">Value:</Text>
              <Text fontSize="xs" color="white" fontWeight="bold">
                {news.value || '0'}
              </Text>
            </HStack>

            <HStack justify="space-between">
              <Text fontSize="xs" color="whiteAlpha.600">Type:</Text>
              <Badge colorScheme="purple" fontSize="xs">
                {news.type || 'Unknown'}
              </Badge>
            </HStack>

            <HStack justify="space-between">
              <Text fontSize="xs" color="whiteAlpha.600">Token:</Text>
              <Badge colorScheme="blue" fontSize="xs">
                {news.token || 'Unknown'}
              </Badge>
            </HStack>

            <Divider borderColor="whiteAlpha.200" />

            <Button
              size="sm"
              variant="ghost"
              colorScheme="orange"
              leftIcon={<CopyIcon />}
              onClick={() => copyToClipboard(JSON.stringify(news, null, 2))}
              width="100%"
            >
              Copy Transaction Data
            </Button>
          </VStack>
        </CardBody>
      </Card>
    );
  }

  return (
    <Card 
      bg="whiteAlpha.100" 
      borderColor="whiteAlpha.200" 
      borderWidth="1px"
      _hover={{ borderColor: 'brand.orange' }}
      transition="all 0.2s"
    >
      <CardBody py={2} px={3}>
        <Box>
          <HStack justify="space-between" mb={1}>
            <Text fontSize="xs" color="white">
              {news.date}
            </Text>
            <HStack spacing={1}>
              <Text
                fontSize="xs"
                color="white"
                cursor="pointer"
                _hover={{ color: 'brand.orange' }}
                onClick={handleSourceClick}
              >
                {news.hacker_group}
              </Text>
              <Text fontSize="xs" color="whiteAlpha.600">
                ({(news.confidence * 100).toFixed(1)}%)
              </Text>
            </HStack>
          </HStack>

          <Text
            fontSize="sm"
            fontWeight="medium"
            mb={2}
            cursor="pointer"
            color="white"
            _hover={{ color: 'brand.orange' }}
            onClick={handleNewsClick}
            lineHeight="1.2"
          >
            {news.title}
          </Text>

          <HStack justify="space-between" align="center" spacing={2}>
            <HStack spacing={2}>
              <Badge colorScheme="red" fontSize="xs" px={1.5} py={0.5}>
                {news.hack_type}
              </Badge>
              <Text color="red.400" fontWeight="bold" fontSize="xs">
                {news.cash_formatted}
              </Text>
            </HStack>
            
            <HStack spacing={2} color="white" fontSize="xs">
              <Text>W:{news.total_words}</Text>
              <Text>B:{news.brutal_words}</Text>
              <Text>{(news.brutal_prop * 100).toFixed(1)}%</Text>
            </HStack>
          </HStack>
        </Box>
      </CardBody>
    </Card>
  );
};

export default NewsCard; 