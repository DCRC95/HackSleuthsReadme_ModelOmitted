import React, { useEffect, useState } from 'react';
import { VStack, Text, Divider, Box, Icon, Link, HStack, IconButton, Badge, Progress, Button, Tooltip, useToast, Flex } from '@chakra-ui/react';
import { FiX, FiExternalLink, FiTrendingUp, FiDownload } from 'react-icons/fi';
import { CopyIcon } from '@chakra-ui/icons';
import { useSource } from '../contexts/SourceContext';
import { useNews } from '../contexts/NewsContext';
import { useNavigate, useLocation } from 'react-router-dom';
import { fetchNetworkVisualizationDetails, fetchNetworkExplanationSummary, fetchNetworkDiagram, fetchReportFile } from '../services/api';


const RightSidebar: React.FC = () => {
  const { selectedSource } = useSource();
  const { selectedNews, setSelectedNews } = useNews();
  const [networkVisualizationImg, setNetworkVisualizationImg] = useState("");
  const [networkDiagramImg, setNetworkDiagramImg] = useState("");
  const [explanationText, setExplanationText] = useState("");
  const location = useLocation();
  const isSars = location.pathname.startsWith("/sars");
  const navigate = useNavigate();
  const toast = useToast();
  const handleClose = () => {
    setSelectedNews(null);
  };

  useEffect(() => {
    if (!isSars && selectedNews?.title) {
      fetchNetworkVisualizationDetails(selectedNews.title).then((data) => {
        setNetworkVisualizationImg(`data:image/png;base64,${data.base64_data}`);
      });

      fetchNetworkDiagram(selectedNews.title).then((data) => {
        setNetworkDiagramImg(`data:image/png;base64,${data.base64_data}`)
      })

      fetchNetworkExplanationSummary(selectedNews.title).then((data) => {
        setExplanationText(data.explanation_text)
      })
    }
  }, [selectedNews, isSars]);


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
    <Box
      bg="brand.gray"
      borderLeft="1px solid"
      borderColor="whiteAlpha.200"
      height="calc(100vh - 60px)"
      position="fixed"
      right="0"
      top="60px"
      width="950px"
      overflowY="auto"
      p={6}
    >
      {selectedNews ? (
        <VStack spacing={4} align="stretch">
          <HStack justify="space-between">
            <Text fontSize="xl" fontWeight="bold">
              {selectedNews.title}
            </Text>
            <IconButton
              aria-label="Close"
              icon={<FiX />}
              variant="ghost"
              onClick={handleClose}
            />
          </HStack>

          <Divider borderColor="whiteAlpha.400" />

          <VStack align="stretch" spacing={3}>
            <HStack justify="space-between">
              <Text color="whiteAlpha.700">Amount Lost:</Text>
              <Text>{selectedNews.cash_formatted}</Text>
            </HStack>

            <HStack justify="space-between">
              <Text color="whiteAlpha.700">Date:</Text>
              <Text>{selectedNews.date}</Text>
            </HStack>

            <HStack justify="space-between">
              <Text color="whiteAlpha.700">Hack Type:</Text>
              <Badge colorScheme={selectedNews.hack_type === "Unknown" ? "gray" : "red"}>
                {selectedNews.hack_type}
              </Badge>
            </HStack>

            <HStack justify="space-between">
              <Text color="whiteAlpha.700">Hacker Group:</Text>
              <Badge colorScheme={selectedNews.hacker_group === "Unknown" ? "gray" : "purple"}>
                {selectedNews.hacker_group}
              </Badge>
            </HStack>

            <HStack justify="space-between">
              <Text color="whiteAlpha.700">Confidence:</Text>
              <Box w="150px">
                <Progress
                  value={selectedNews.confidence * 100}
                  colorScheme={selectedNews.confidence > 0.5 ? "green" : "orange"}
                  borderRadius="full"
                />
              </Box>
            </HStack>

            <HStack justify="space-between">
              <Text color="whiteAlpha.700">Hash:</Text>
              <Tooltip label="Click to copy" hasArrow>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => copyToClipboard(String(selectedNews.hash))}
                  color="whiteAlpha"
                  _hover={{ color: 'brand.orange' }}
                  leftIcon={<CopyIcon />}
                >
                  <Text>{selectedNews.hash}</Text>
                </Button>
              </Tooltip>

            </HStack>

            <Flex justifyContent={'space-between'}>
              <Button
                leftIcon={<FiTrendingUp />}
                colorScheme="blue"
                variant="solid"
                onClick={() => navigate(`/sars/${selectedNews.title}`)}
                w="49%"
                mt={2}
              >
                SARS
              </Button>

              <Button
                leftIcon={<FiDownload />}
                colorScheme="blue"
                variant="solid"
                onClick={() => fetchReportFile(selectedNews.title)}
                w="49%"
                mt={2}
              >
                Download Report
              </Button>
            </Flex>

            <Flex>
              <Box mt={2} maxW="md">
                <Text color="whiteAlpha.700" mb={2}>Full Story:</Text>
                <Box
                  bg="whiteAlpha.100"
                  p={2}
                  borderRadius="md"
                  maxH="400px"
                  overflowY="auto"
                >
                  <Text whiteSpace="pre-wrap">
                    {selectedNews.content || "No detailed story available for this hack."}
                  </Text>
                </Box>
              </Box>
              <Box mt={2} maxW="md">
                <Text color="whiteAlpha.700" mb={2}>Explanation:</Text>
                <Box
                  bg="whiteAlpha.100"
                  p={2}
                  borderRadius="md"
                  maxH="400px"
                  overflowY="auto"
                >
                  <Text whiteSpace="pre-wrap">
                    {explanationText || "No detailed story available for this hack."}
                  </Text>
                </Box>
              </Box>
            </Flex>

            <Flex>
              <Box mt={4}>
                <Text color="whiteAlpha.700" mb={2}>Network Visualisation:</Text>
                <Box
                  bg="whiteAlpha.100"
                  p={4}
                  borderRadius="md"
                >
                  <img src={networkVisualizationImg} alt="Network Visualization" />
                </Box>
              </Box>
              <Box mt={4}>
                <Text color="whiteAlpha.700" mb={2}>Offshore Diagram:</Text>
                <Box
                  bg="whiteAlpha.100"
                  p={4}
                  borderRadius="md"
                >
                  <img src={networkDiagramImg} alt="Network Diagram" />
                </Box>
              </Box>
            </Flex>
          </VStack>
        </VStack>
      ) : selectedSource ? (
        <VStack spacing={4} align="stretch">
          <HStack justify="space-between">
            <Text fontSize="xl" fontWeight="bold">
              {selectedSource.name}
            </Text>
            <IconButton
              aria-label="Close"
              icon={<FiX />}
              variant="ghost"
              onClick={() => setSelectedNews(null)}
            />
          </HStack>

          <Divider borderColor="whiteAlpha.400" />

          <Text>{selectedSource.description}</Text>

          {selectedSource.url && (
            <Link href={selectedSource.url} isExternal color="blue.400">
              <HStack>
                <Text>Visit Website</Text>
                <Icon as={FiExternalLink} />
              </HStack>
            </Link>
          )}
        </VStack>
      ) : (
        <Text color="whiteAlpha.600" textAlign="center">
          Select a news item or source to view details
        </Text>
      )}
    </Box>
  );
};

export default RightSidebar; 