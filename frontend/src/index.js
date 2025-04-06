import React from 'react';
import ReactDOM from 'react-dom/client';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import './index.css';
import './styles.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

// Create a theme instance with neo-modern black and green colors
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00E676', // Bright green
      light: '#69F0AE',
      dark: '#00C853',
      contrastText: '#000',
    },
    secondary: {
      main: '#1DE9B6', // Teal-ish green
      light: '#64FFDA',
      dark: '#00BFA5',
    },
    background: {
      default: '#121212', // Very dark gray, almost black
      paper: '#1E1E1E',   // Slightly lighter dark gray
    },
    text: {
      primary: '#FFFFFF',
      secondary: '#B0BEC5',
    },
    success: {
      main: '#00E676',
    },
    error: {
      main: '#FF5252',
    },
    divider: 'rgba(0, 230, 118, 0.12)',
  },
  typography: {
    fontFamily: [
      'Roboto',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
      '"Apple Color Emoji"',
      '"Segoe UI Emoji"',
      '"Segoe UI Symbol"',
    ].join(','),
    h1: { fontWeight: 500 },
    h2: { fontWeight: 500 },
    h3: { fontWeight: 500 },
    h4: { fontWeight: 500 },
    h5: { fontWeight: 500 },
    h6: { fontWeight: 500 },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
        containedPrimary: {
          '&:hover': {
            boxShadow: '0 0 12px rgba(0, 230, 118, 0.8)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          borderRadius: 12,
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: '#1A1A1A',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.5)',
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          borderBottom: '1px solid rgba(0, 230, 118, 0.12)',
        },
        head: {
          fontWeight: 700,
          color: '#00E676',
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          '&:hover': {
            backgroundColor: 'rgba(0, 230, 118, 0.04) !important',
          },
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        standardSuccess: {
          backgroundColor: 'rgba(0, 230, 118, 0.15)',
          color: '#69F0AE',
        },
        standardInfo: {
          backgroundColor: 'rgba(29, 233, 182, 0.15)',
          color: '#64FFDA',
        },
      },
    },
  },
});

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
