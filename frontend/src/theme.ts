import { extendTheme } from '@chakra-ui/react';

const theme = extendTheme({
  styles: {
    global: {
      body: {
        bg: 'brand.background',
        color: 'white',
      },
    },
  },
  colors: {
    brand: {
      background: '#1C1C28',
      gray: '#23232F',
      orange: '#FF3B3B',
      hover: {
        orange: '#FF5252',
      },
    },
  },
  components: {
    Button: {
      variants: {
        ghost: {
          _hover: {
            bg: 'whiteAlpha.100',
          },
        },
      },
    },
    Link: {
      baseStyle: {
        _hover: {
          textDecoration: 'none',
        },
      },
    },
  },
  config: {
    initialColorMode: 'dark',
    useSystemColorMode: false,
  },
});

export default theme; 