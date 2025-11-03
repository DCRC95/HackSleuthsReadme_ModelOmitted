import React, { useEffect, useState } from 'react';
import { VStack, Box, Spinner, Text, Center } from '@chakra-ui/react';
import { useLocation } from 'react-router-dom';
import NewsCard from './NewsCard';
import { NewsItem } from '../types';
import { fetchHackAnalysisData, fetchSarsList, fetchSarsFilteredList } from '../services/api';
import { useParams } from 'react-router-dom';
import { useNews } from '../contexts/NewsContext';

const MainContent: React.FC = () => {
  const location = useLocation();
  const isSarsRoute = location.pathname.startsWith('/sars');
  const { title } = useParams();
  const [newsData, setNewsData] = useState<NewsItem[]>([]);
  const { selectedNews } = useNews();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setIsLoading(true);
        let data: any;
        if (isSarsRoute) {
          data = title ? await fetchSarsFilteredList(title) : await fetchSarsList()
          setNewsData([])

          for (let i = 0; i < data.length; i++) {
            if (data[i]) {
              const element = data[i];
              setTimeout(() => {
                setNewsData((prevData)=>[...prevData, element])
              }, i)
            }
          }
        } else {
          data = await fetchHackAnalysisData()
          setNewsData(data);
        }
        setError(null);
      } catch (err) {
        setError('Failed to load hack analysis data. Please try again later.');
        console.error('Error loading data:', err);
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, [isSarsRoute, title]); // Re-fetch when route changes

  if (isLoading) {
    return (
      <Center h="calc(100vh - 60px)" ml="240px" mr={isSarsRoute ? "0" : "400px"}>
        <Spinner size="xl" color="brand.orange" />
      </Center>
    );
  }

  if (error) {
    return (
      <Center h="calc(100vh - 60px)" ml="240px" mr={isSarsRoute ? "0" : "400px"}>
        <Text color="red.400">{error}</Text>
      </Center>
    );
  }

  return (
    <Box
      maxW={selectedNews ? "500px" : "800px"}
      mx="auto"
      mt="25px"
      pt="60px"
      px={6}
      pb={6}
      ml="240px" // LeftSidebar width
      mr={isSarsRoute ? "0" : "650px"} // RightSidebar width only when not in SARS route
    >
      <VStack spacing={4} align="stretch">
        {newsData.map((news, index) => (
          <NewsCard key={index} news={news} isSarsRoute={isSarsRoute} />
        ))}
      </VStack>
    </Box>
  );
};

export default MainContent; 