import React, { useState, useEffect, createContext, useContext } from 'react';
import { 
  Box, 
  AppBar,
  Toolbar,
  IconButton,
  Typography,
  Fab,
  useTheme,
  LinearProgress,
  Modal,
  Backdrop,
  Fade,
  Button,
  Divider
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { motion } from 'framer-motion';
import MenuIcon from '@mui/icons-material/Menu';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import CloseIcon from '@mui/icons-material/Close';

// Components
import Sidebar from './components/Sidebar';
import Home from './components/Home';
import DataExplorer from './components/DataExplorer';
import Analysis from './components/Analysis';
import DataGeneration from './components/DataGeneration';
import Database from './components/Database';
import Acknowledgements from './components/Acknowledgements';
import About from './components/About';

// API service
import api from './services/api';

// Calculate the width for main content with sidebar
const drawerWidth = 280;

const Main = styled('main', { shouldForwardProp: (prop) => prop !== 'open' })(
  ({ theme, open }) => ({
    flexGrow: 1,
    padding: 0,
    transition: theme.transitions.create(['width', 'margin'], {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.leavingScreen,
    }),
    marginLeft: `${drawerWidth}px`,
    marginRight: 0, // Remove right margin for more space
    width: `calc(100% - ${drawerWidth}px)`, // Only subtract sidebar width, not double
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start', 
    overflowX: 'hidden',
    position: 'relative',
    alignItems: 'flex-start',
  }),
);

// Styled component for app bar
const StyledAppBar = styled(AppBar, { shouldForwardProp: (prop) => prop !== 'open' })(
  ({ theme, open }) => ({
    transition: theme.transitions.create(['margin', 'width'], {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.leavingScreen,
    }),
    background: 'linear-gradient(90deg, #121212, #1E1E1E)',
    borderBottom: '1px solid rgba(0, 230, 118, 0.2)',
    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5)',
    ...(open && {
      width: `calc(100% - ${drawerWidth}px)`,
      marginLeft: `${drawerWidth}px`,
      marginRight: 0, // Remove right margin for more space
      transition: theme.transitions.create(['margin', 'width'], {
        easing: theme.transitions.easing.easeOut,
        duration: theme.transitions.duration.enteringScreen,
      }),
    }),
  }),
);

// Styled component for toolbar spacing
const DrawerHeader = styled('div')(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  padding: theme.spacing(0, 1),
  ...theme.mixins.toolbar,
  justifyContent: 'flex-end',
}));

// Create context for CSV upload state
export const CSVContext = createContext({
  isCSVUploaded: false,
  setCSVUploaded: () => {}
});

// Create context for generation status
export const GenerationContext = createContext({
  generationStatus: {
    isGenerating: false,
    progress: 0,
    currentFile: null,
    error: null
  },
  setGenerationStatus: () => {}
});

// Create context for upload response
export const UploadResponseContext = createContext({
  uploadResponse: null,
  setUploadResponse: () => {}
});

function App() {
  const theme = useTheme();
  
  // State management
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [file, setFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResponse, setUploadResponse] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [currentQuery, setCurrentQuery] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);
  const [streamedResponse, setStreamedResponse] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  
  // Generation status state
  const [generationStatus, setGenerationStatus] = useState({
    isGenerating: false,
    progress: 0,
    currentFile: null,
    error: null
  });
  
  // CSV upload context state
  const [isCSVUploaded, setCSVUploaded] = useState(false);
  
  // Help modal state
  const [helpOpen, setHelpOpen] = useState(false);
  
  // Toggle sidebar
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };
  
  // Check if we have a file uploaded on initialization
  useEffect(() => {
    const savedState = localStorage.getItem('csv_uploaded');
    if (savedState === 'true') {
      setCSVUploaded(true);
      
      // Also check the server for current CSV status and update uploadResponse
      const fetchCSVStatus = async () => {
        try {
          const response = await api.checkCSVStatus();
          if (response && response.has_csv) {
            setUploadResponse(response);
          }
        } catch (error) {
          console.error('Error checking CSV status on app initialization:', error);
        }
      };
      
      fetchCSVStatus();
    }
  }, []);
  
  // Update localStorage when upload state changes
  useEffect(() => {
    localStorage.setItem('csv_uploaded', isCSVUploaded);
  }, [isCSVUploaded]);
  
  // Set CSV uploaded to true when we receive upload response
  useEffect(() => {
    if (uploadResponse && uploadResponse.success) {
      setCSVUploaded(true);
    }
  }, [uploadResponse]);

  // Handle file selection
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    
    if (selectedFile) {
      // Read file for preview
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const csv = e.target.result;
          const lines = csv.split('\n');
          const headers = lines[0].split(',');
          
          // Create a preview of the data
          const preview = [];
          for (let i = 1; i < Math.min(lines.length, 11); i++) {
            if (lines[i].trim()) {
              const values = lines[i].split(',');
              const row = {};
              headers.forEach((header, index) => {
                row[header.trim()] = values[index] ? values[index].trim() : '';
              });
              preview.push(row);
            }
          }
          
          setFilePreview({
            headers,
            preview,
            rowCount: lines.length - 1,
            columnCount: headers.length
          });
        } catch (error) {
          console.error('Error parsing CSV:', error);
        }
      };
      reader.readAsText(selectedFile);
    }
  };

  // Handle file upload
  const handleUpload = async () => {
    if (!file) return;
    
    setIsUploading(true);
    
    try {
      const data = await api.uploadFile(file);
      setUploadResponse(data);
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadResponse({
        error: error.response?.data?.error || 'Failed to upload file'
      });
    } finally {
      setIsUploading(false);
    }
  };

  // Handle sending a chat query
  const handleSendQuery = async () => {
    if (!currentQuery.trim()) return;
    
    // Add user message to chat history
    const newMessage = { sender: 'user', message: currentQuery };
    setChatHistory(prev => [...prev, newMessage]);
    setIsQuerying(true);
    setStatusMessage('Analyzing your question...');
    
    try {
      // Set up for streaming response
      setStreamedResponse('');
      
      const response = await api.streamAnalysis(currentQuery);
      
      // Handle the streaming response
      await api.handleStreamResponse(
        response,
        // Content callback
        (fullText, chunk) => {
          setStreamedResponse(fullText);
        },
        // Info callback
        (message) => {
          setStatusMessage(message);
        },
        // Error callback
        (errorMessage) => {
          setStatusMessage(`Error: ${errorMessage}`);
        },
        // Complete callback
        (message) => {
          setStatusMessage(message);
        }
      ).then(fullResponse => {
        // Add assistant response to chat history
        if (fullResponse) {
          setChatHistory(prev => [...prev, { 
            sender: 'assistant', 
            message: fullResponse 
          }]);
        }
      });
      
    } catch (error) {
      console.error('Error sending query:', error);
      setChatHistory(prev => [...prev, { 
        sender: 'assistant', 
        message: 'Sorry, there was an error processing your request.' 
      }]);
    } finally {
      setIsQuerying(false);
      setCurrentQuery('');
    }
  };

  return (
    <CSVContext.Provider value={{ isCSVUploaded, setCSVUploaded }}>
      <GenerationContext.Provider value={{ generationStatus, setGenerationStatus }}>
        <UploadResponseContext.Provider value={{ uploadResponse, setUploadResponse }}>
          <Router>
            <Box sx={{ 
              display: 'flex',
              width: '100%',
              padding: 0,
              m: 0,
              overflow: 'hidden',
              height: '100vh'
            }} className="matrix-bg">
              <StyledAppBar position="fixed" open={sidebarOpen}>
                <Toolbar>
                  <IconButton
                    color="inherit"
                    aria-label="open drawer"
                    onClick={toggleSidebar}
                    edge="start"
                    sx={{ mr: 2, ...(sidebarOpen && { display: 'none' }) }}
                    className="neon-button"
                  >
                    <MenuIcon />
                  </IconButton>
                  <Typography 
                    variant="h6" 
                    noWrap 
                    component="div"
                    sx={{ 
                      flexGrow: 1,
                      fontWeight: 'bold',
                      letterSpacing: '2px'
                    }}
                    className="cyber-header"
                  >
                    CSV Upload and Analysis with LlamaIndex
                  </Typography>
                  
                  {/* Generation Progress indicator in top bar */}
                  {uploadResponse && generationStatus.isGenerating && (
                    <Box sx={{ display: 'flex', alignItems: 'center', ml: 2, maxWidth: 300 }}>
                      <Box sx={{ width: '100%', mr: 1 }}>
                        <LinearProgress 
                          variant="determinate" 
                          value={generationStatus.progress} 
                          sx={{ 
                            height: 8, 
                            borderRadius: 4,
                            '& .MuiLinearProgress-bar': {
                              background: 'linear-gradient(90deg, #00C853, #00E676, #69F0AE)',
                            }
                          }}
                        />
                      </Box>
                      <Box sx={{ minWidth: 35 }}>
                        <Typography variant="body2" color="white">
                          {`${Math.round(generationStatus.progress)}%`}
                        </Typography>
                      </Box>
                    </Box>
                  )}
                </Toolbar>
              </StyledAppBar>
              
              {/* Sidebar without generation progress */}
              <Sidebar 
                isOpen={sidebarOpen} 
                toggleSidebar={toggleSidebar} 
                uploadResponse={uploadResponse}
                isGenerating={generationStatus.isGenerating}
                generationProgress={generationStatus.progress}
              />
              
              {/* Main content */}
              <Main open={sidebarOpen}>
                <DrawerHeader />
                
                {/* Routes */}
                <Box sx={{ 
                  width: '100%', 
                  maxWidth: '100%',
                  m: 0,
                  p: 3,
                  boxSizing: 'border-box',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'stretch',
                  flexGrow: 1,
                  minHeight: 'calc(100vh - 64px)', // Minimum height instead of fixed height
                  overflow: 'auto',
                  scrollBehavior: 'smooth',
                  '&::-webkit-scrollbar': {
                    width: '8px'
                  },
                  '&::-webkit-scrollbar-track': {
                    background: 'rgba(0, 0, 0, 0.1)'
                  },
                  '&::-webkit-scrollbar-thumb': {
                    background: 'rgba(0, 230, 118, 0.3)',
                    borderRadius: '4px'
                  },
                  '&::-webkit-scrollbar-thumb:hover': {
                    background: 'rgba(0, 230, 118, 0.5)'
                  },
                  position: 'relative',
                  zIndex: 1
                }}>
                  <Routes>
                    <Route 
                      path="/" 
                      element={
                        <Home 
                          file={file}
                          setFile={setFile}
                          filePreview={filePreview}
                          setFilePreview={setFilePreview}
                          isUploading={isUploading}
                          setIsUploading={setIsUploading}
                          uploadResponse={uploadResponse}
                          setUploadResponse={setUploadResponse}
                          chatHistory={chatHistory}
                          setChatHistory={setChatHistory}
                          currentQuery={currentQuery}
                          setCurrentQuery={setCurrentQuery}
                          isQuerying={isQuerying}
                          setIsQuerying={setIsQuerying}
                          streamedResponse={streamedResponse}
                          setStreamedResponse={setStreamedResponse}
                          statusMessage={statusMessage}
                          setStatusMessage={setStatusMessage}
                          handleFileChange={handleFileChange}
                          handleUpload={handleUpload}
                          handleSendQuery={handleSendQuery}
                          generationStatus={generationStatus}
                        />
                      }
                    />
                    <Route path="/explorer" element={<DataExplorer />} />
                    <Route path="/analysis" element={<Analysis />} />
                    <Route path="/generation" element={<DataGeneration />} />
                    <Route path="/database" element={<Database />} />
                    <Route path="/acknowledgements" element={<Acknowledgements />} />
                    <Route path="/about" element={<About />} />
                  </Routes>
                </Box>
                
                {/* Floating action button */}
                <Fab 
                  aria-label="help" 
                  sx={{ 
                    position: 'fixed', 
                    bottom: 20, 
                    right: 20,
                    bgcolor: 'rgba(0, 230, 118, 0.1)',
                    color: 'rgba(0, 230, 118, 0.9)',
                    border: '1px solid rgba(0, 230, 118, 0.3)',
                    '&:hover': {
                      bgcolor: 'rgba(0, 230, 118, 0.2)',
                    }
                  }}
                  className="glow-effect"
                  onClick={() => setHelpOpen(true)}
                >
                  <HelpOutlineIcon />
                </Fab>
                
                {/* Help Modal */}
                <Modal
                  open={helpOpen}
                  onClose={() => setHelpOpen(false)}
                  closeAfterTransition
                  BackdropComponent={Backdrop}
                  BackdropProps={{
                    timeout: 500,
                  }}
                >
                  <Fade in={helpOpen}>
                    <Box sx={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      width: '80%',
                      maxWidth: 800,
                      maxHeight: '80vh',
                      overflow: 'auto',
                      bgcolor: 'rgba(0, 0, 0, 0.9)',
                      border: '1px solid rgba(0, 230, 118, 0.5)',
                      boxShadow: '0 4px 30px rgba(0, 230, 118, 0.3)',
                      p: 4,
                      borderRadius: 2,
                    }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                        <Typography variant="h5" component="h2" sx={{ color: '#00E676' }}>
                          Application Help Guide
                        </Typography>
                        <IconButton 
                          onClick={() => setHelpOpen(false)}
                          sx={{ color: 'rgba(255, 255, 255, 0.7)' }}
                        >
                          <CloseIcon />
                        </IconButton>
                      </Box>
                      
                      <Typography variant="h6" sx={{ color: '#00E676', mt: 3, mb: 1 }}>
                        Getting Started
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)', mb: 2 }}>
                        This application allows you to upload CSV files, analyze data, generate synthetic data, and explore insights using large language models.
                      </Typography>
                      
                      <Divider sx={{ my: 2, borderColor: 'rgba(0, 230, 118, 0.2)' }} />
                      
                      <Typography variant="h6" sx={{ color: '#00E676', mt: 3, mb: 1 }}>
                        Main Features
                      </Typography>
                      
                      <Typography variant="subtitle2" sx={{ color: '#00E676', mt: 2, mb: 0.5 }}>
                        Home
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)', mb: 2 }}>
                        The main dashboard where you can upload CSV files and ask questions about your data.
                      </Typography>
                      
                      <Typography variant="subtitle2" sx={{ color: '#00E676', mt: 2, mb: 0.5 }}>
                        Data Explorer
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)', mb: 2 }}>
                        Visualize your data with interactive charts and statistics to better understand its patterns and distributions.
                      </Typography>
                      
                      <Typography variant="subtitle2" sx={{ color: '#00E676', mt: 2, mb: 0.5 }}>
                        Analysis
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)', mb: 2 }}>
                        Get insights from your data by asking questions in natural language. The application uses AI to analyze your data and provide meaningful answers.
                      </Typography>
                      
                      <Typography variant="subtitle2" sx={{ color: '#00E676', mt: 2, mb: 0.5 }}>
                        Data Generation
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)', mb: 2 }}>
                        Generate synthetic data based on your original dataset. This feature creates new data points that maintain the statistical properties and relationships of your original data.
                      </Typography>
                      
                      <Typography variant="subtitle2" sx={{ color: '#00E676', mt: 2, mb: 0.5 }}>
                        Embedding Visualization
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)', mb: 2 }}>
                        Explore semantic similarities between different parts of your data using vector embeddings. This helps identify patterns and relationships that might not be immediately obvious.
                      </Typography>
                      
                      <Divider sx={{ my: 2, borderColor: 'rgba(0, 230, 118, 0.2)' }} />
                      
                      <Typography variant="h6" sx={{ color: '#00E676', mt: 3, mb: 1 }}>
                        Workflow
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)', mb: 0.5 }}>
                        1. Upload a CSV file from the Home page
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)', mb: 0.5 }}>
                        2. Explore your data in the Data Explorer
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)', mb: 0.5 }}>
                        3. Ask questions and analyze your data in the Analysis section
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)', mb: 0.5 }}>
                        4. Generate synthetic data in the Data Generation section
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)', mb: 3 }}>
                        5. Visualize embeddings to understand semantic relationships
                      </Typography>
                      
                      <Button
                        variant="outlined"
                        onClick={() => setHelpOpen(false)}
                        sx={{ 
                          display: 'block',
                          mx: 'auto',
                          mt: 3,
                          color: '#00E676',
                          borderColor: 'rgba(0, 230, 118, 0.5)',
                          '&:hover': {
                            borderColor: '#00E676',
                            bgcolor: 'rgba(0, 230, 118, 0.1)'
                          }
                        }}
                      >
                        Got it
                      </Button>
                    </Box>
                  </Fade>
                </Modal>
              </Main>
            </Box>
          </Router>
        </UploadResponseContext.Provider>
      </GenerationContext.Provider>
    </CSVContext.Provider>
  );
}

export default App;
