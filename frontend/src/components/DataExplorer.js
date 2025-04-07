import React, { useState, useEffect, useContext } from 'react';
import { 
  Paper, 
  Typography, 
  Box, 
  Button, 
  Grid, 
  TableContainer, 
  Table, 
  TableHead, 
  TableBody, 
  TableRow, 
  TableCell,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress,
  Alert,
  useTheme
} from '@mui/material';
import { motion } from 'framer-motion';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import DownloadIcon from '@mui/icons-material/Download';
import DeleteIcon from '@mui/icons-material/Delete';
import api from '../services/api';
import QueryInterface from './QueryInterface';
import { CSVContext, UploadResponseContext } from '../App';

const DataExplorer = () => {
  const theme = useTheme();
  const [file, setFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [localUploadResponse, setLocalUploadResponse] = useState(null);
  const [isFullPreview, setIsFullPreview] = useState(false);
  
  // Access global context
  const { setCSVUploaded } = useContext(CSVContext);
  const { uploadResponse, setUploadResponse } = useContext(UploadResponseContext);
  
  // Load persisted data on component mount
  useEffect(() => {
    const checkForCurrentCSV = async () => {
      try {
        // Check if there's a current CSV file on the server
        const response = await api.checkCSVStatus();
        
        if (response.hasCSV || response.has_csv) {
          // We have a CSV file, create a response object
          const responseData = {
            success: true,
            message: `Using uploaded file: ${response.filename || response.current_file}`,
            rows: response.rows || response.rowCount,
            columns: response.columnNames || response.columns,
            filename: response.filename || (response.current_file ? response.current_file.split('/').pop() : 'dataset.csv'),
            rowCount: response.rows || response.rowCount,
            columnCount: (response.columnNames ? response.columnNames.length : 0) || 
                        (response.columns ? response.columns.length : 0) || 0,
            has_csv: true
          };
          
          // Update both local and global state
          setLocalUploadResponse(responseData);
          setUploadResponse(responseData);
          setCSVUploaded(true);
          
          // No need to update filePreview unless user re-uploads
          // This prevents stale preview data from being shown
          if (file) {
            // Only if user has selected a file in this session
            const reader = new FileReader();
            reader.onload = handleFileRead;
            reader.readAsText(file);
          }
        } else {
          // No CSV file, clear any cached data
          setFilePreview(null);
          localStorage.removeItem('filePreview');
          localStorage.removeItem('uploadResponse');
        }
      } catch (error) {
        console.error('Error checking CSV status:', error);
      }
    };
    
    checkForCurrentCSV();
  }, [setUploadResponse, setCSVUploaded]);
  
  // File reader callback function
  const handleFileRead = (e) => {
    try {
      const csv = e.target.result;
      const lines = csv.split('\n');
      const headers = lines[0].split(',');
      
      // Create a preview of the data
      const preview = [];
      const previewLines = isFullPreview ? Math.min(lines.length, 100) : Math.min(lines.length, 11);
      
      for (let i = 1; i < previewLines; i++) {
        if (lines[i].trim()) {
          const values = lines[i].split(',');
          const row = {};
          headers.forEach((header, index) => {
            row[header.trim()] = values[index] ? values[index].trim() : '';
          });
          preview.push(row);
        }
      }
      
      const newPreview = {
        headers,
        preview,
        rowCount: lines.length - 1,
        columnCount: headers.length,
        totalLines: lines.length
      };
      
      setFilePreview(newPreview);
    } catch (error) {
      console.error('Error parsing CSV:', error);
    }
  };
  
  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        duration: 0.5,
        when: 'beforeChildren',
        staggerChildren: 0.2
      }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.3 }
    }
  };
  
  // Handle file selection
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    
    if (selectedFile) {
      // Read file for preview
      const reader = new FileReader();
      reader.onload = handleFileRead;
      reader.readAsText(selectedFile);
    }
  };

  // Handle file upload
  const handleUpload = async () => {
    if (!file) return;
    
    setIsUploading(true);
    
    try {
      const data = await api.uploadFile(file);
      const response = {
        ...data,
        message: 'File uploaded successfully! Go to the Data Generation tab to generate synthetic data.'
      };
      
      // Update both local and global state
      setLocalUploadResponse(response);
      setUploadResponse(response);
      setCSVUploaded(true);
    } catch (error) {
      console.error('Error uploading file:', error);
      const errorResponse = {
        error: error.response?.data?.error || 'Failed to upload file'
      };
      setLocalUploadResponse(errorResponse);
    } finally {
      setIsUploading(false);
    }
  };
  
  // Handle data deletion
  const handleDeleteData = async () => {
    setIsDeleting(true);
    
    try {
      const response = await api.deleteCurrentCSV();
      
      if (response.success) {
        // Clear states
        setFile(null);
        setFilePreview(null);
        
        // Show success message in local state
        setLocalUploadResponse({
          message: 'Data deleted successfully.',
          success: true
        });
        
        // Update global state
        setUploadResponse(null);
        setCSVUploaded(false);
      } else {
        setLocalUploadResponse({
          error: response.message || 'Failed to delete data'
        });
      }
    } catch (error) {
      console.error('Error deleting data:', error);
      setLocalUploadResponse({
        error: error.message || 'Failed to delete data'
      });
    } finally {
      setIsDeleting(false);
    }
  };
  
  // Toggle full preview
  const toggleFullPreview = () => {
    setIsFullPreview(!isFullPreview);
    if (file) {
      // Re-trigger file reading with new preview size
      handleFileChange({ target: { files: [file] } });
    }
  };
  
  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <Paper 
        sx={{ 
          p: 3, 
          mb: 3,
          border: '1px solid rgba(0, 230, 118, 0.2)',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4)',
        }}
      >
        <Typography 
          variant="h4" 
          component="h1" 
          gutterBottom
          className="cyber-header"
          sx={{ mb: 3 }}
        >
          Data Explorer
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <motion.div variants={itemVariants}>
              <Paper 
                sx={{ 
                  p: 3, 
                  mb: 3,
                  border: '1px solid rgba(0, 230, 118, 0.2)',
                  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4)',
                }}
              >
                <Typography 
                  variant="h5" 
                  component="h2" 
                  gutterBottom
                  className="cyber-header"
                  sx={{ mb: 3 }}
                >
                  Upload & Explore CSV File
                </Typography>
                
                <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Box>
                    <Button
                      variant="contained"
                      component="label"
                      startIcon={<CloudUploadIcon />}
                      sx={{ 
                        mb: 2,
                        borderRadius: '8px',
                        px: 3,
                        py: 1.2,
                      }}
                      className="glow-effect"
                    >
                      Choose File
                      <input
                        type="file"
                        accept=".csv"
                        hidden
                        onChange={handleFileChange}
                      />
                    </Button>
                    {file && (
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          ml: 2,
                          display: 'inline-block',
                          px: 2,
                          py: 0.5,
                          borderRadius: '4px',
                          bgcolor: 'rgba(0, 230, 118, 0.1)',
                          border: '1px solid rgba(0, 230, 118, 0.3)'
                        }}
                      >
                        {file.name} ({Math.round(file.size / 1024)} KB)
                      </Typography>
                    )}
                  </Box>
                  
                  <Button
                    variant="outlined"
                    color="error"
                    startIcon={<DeleteIcon />}
                    disabled={isDeleting}
                    onClick={handleDeleteData}
                    sx={{ 
                      borderRadius: '8px',
                      borderColor: 'rgba(255, 82, 82, 0.3)',
                      '&:hover': {
                        borderColor: 'rgba(255, 82, 82, 0.8)',
                        backgroundColor: 'rgba(255, 82, 82, 0.1)',
                      }
                    }}
                  >
                    {isDeleting ? 'Deleting...' : 'Delete Data'}
                  </Button>
                </Box>
                
                {filePreview && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                  >
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Typography 
                        variant="h6" 
                        gutterBottom
                        sx={{ 
                          color: theme.palette.primary.light,
                          borderBottom: '2px solid rgba(0, 230, 118, 0.3)',
                          pb: 1,
                          mb: 0
                        }}
                      >
                        Preview of Data
                      </Typography>
                      
                      <Button 
                        variant="outlined" 
                        size="small"
                        onClick={toggleFullPreview}
                        sx={{ 
                          borderColor: 'rgba(0, 230, 118, 0.3)',
                          color: theme.palette.primary.light
                        }}
                      >
                        {isFullPreview ? 'Show Less' : 'Show More'}
                      </Button>
                    </Box>
                    
                    <TableContainer 
                      component={Paper} 
                      sx={{ 
                        maxHeight: isFullPreview ? 600 : 300, 
                        mb: 3,
                        border: '1px solid rgba(0, 230, 118, 0.2)',
                        borderRadius: '8px',
                        overflow: 'auto',
                        '& .MuiTableCell-head': {
                          bgcolor: 'rgba(0, 230, 118, 0.1)',
                          fontWeight: 'bold',
                          position: 'sticky',
                          top: 0,
                          zIndex: 10
                        }
                      }}
                      className="data-table-container"
                    >
                      <Table stickyHeader size="small">
                        <TableHead className="data-table-header">
                          <TableRow>
                            <TableCell sx={{ bgcolor: 'rgba(0, 0, 0, 0.5)' }}>#</TableCell>
                            {filePreview.headers.map((header, index) => (
                              <TableCell key={index}>{header}</TableCell>
                            ))}
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {filePreview.preview.map((row, rowIndex) => (
                            <TableRow key={rowIndex} hover>
                              <TableCell sx={{ bgcolor: 'rgba(0, 0, 0, 0.3)' }}>{rowIndex + 1}</TableCell>
                              {filePreview.headers.map((header, colIndex) => (
                                <TableCell key={colIndex}>
                                  {row[header] || ''}
                                </TableCell>
                              ))}
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                    
                    <Box 
                      sx={{ 
                        mb: 3, 
                        p: 2, 
                        borderRadius: '8px',
                        border: '1px solid rgba(0, 230, 118, 0.2)', 
                        bgcolor: 'rgba(0, 0, 0, 0.2)'
                      }}
                    >
                      <Typography 
                        variant="h6" 
                        gutterBottom
                        sx={{ color: theme.palette.primary.light }}
                      >
                        Data Statistics
                      </Typography>
                      <Grid container spacing={2} sx={{ mb: 3 }}>
                        <Grid item xs={6} sm={3}>
                          <Paper 
                            sx={{ 
                              p: 2, 
                              textAlign: 'center',
                              border: '1px solid rgba(0, 230, 118, 0.3)',
                              bgcolor: 'rgba(0, 230, 118, 0.05)'
                            }}
                          >
                            <Typography variant="h4" color="primary">
                              {filePreview?.rowCount || (localUploadResponse?.rows || localUploadResponse?.rowCount) || 0}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">Rows</Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={6} md={3}>
                          <Paper 
                            sx={{ 
                              p: 2, 
                              textAlign: 'center',
                              border: '1px solid rgba(0, 230, 118, 0.3)',
                              bgcolor: 'rgba(0, 230, 118, 0.05)'
                            }}
                          >
                            <Typography variant="h4" color="primary">
                              {filePreview?.columnCount || 
                                (localUploadResponse?.columnCount || 
                                 (localUploadResponse?.columns ? localUploadResponse.columns.length : 0)) || 
                                 0}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">Columns</Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={6} md={3}>
                          <Paper 
                            sx={{ 
                              p: 2, 
                              textAlign: 'center',
                              border: '1px solid rgba(0, 230, 118, 0.3)',
                              bgcolor: 'rgba(0, 230, 118, 0.05)'
                            }}
                          >
                            <Typography variant="h4" color="primary">
                              {filePreview ? filePreview.preview.length : 
                               Math.min(localUploadResponse?.rows || localUploadResponse?.rowCount || 0, 10)}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">Preview Rows</Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={6} md={3}>
                          <Paper 
                            sx={{ 
                              p: 2, 
                              textAlign: 'center',
                              border: '1px solid rgba(0, 230, 118, 0.3)',
                              bgcolor: 'rgba(0, 230, 118, 0.05)'
                            }}
                          >
                            <Typography variant="h4" color="primary">
                              {filePreview?.totalLines || 
                               ((localUploadResponse?.rows || localUploadResponse?.rowCount || 0) + 1)}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">Total Lines</Typography>
                          </Paper>
                        </Grid>
                      </Grid>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Button 
                        variant="contained" 
                        color="primary" 
                        onClick={handleUpload} 
                        disabled={isUploading}
                        sx={{ 
                          mt: 2,
                          borderRadius: '8px',
                          px: 3,
                          py: 1.2,
                        }}
                        className="glow-effect"
                        startIcon={<UploadFileIcon />}
                      >
                        {isUploading ? 'Uploading...' : 'Upload File'}
                      </Button>
                      
                      <Button 
                        variant="outlined" 
                        color="primary" 
                        sx={{ 
                          mt: 2,
                          borderRadius: '8px',
                          px: 3,
                          py: 1.2,
                          borderColor: 'rgba(0, 230, 118, 0.3)',
                        }}
                        className="glow-effect-subtle"
                        startIcon={<DownloadIcon />}
                      >
                        Export Data
                      </Button>
                    </Box>
                  </motion.div>
                )}
                
                {isUploading && (
                  <Box sx={{ mt: 3 }}>
                    <LinearProgress 
                      sx={{ 
                        height: 8, 
                        borderRadius: 4,
                        '& .MuiLinearProgress-bar': {
                          background: 'linear-gradient(90deg, #00C853, #00E676, #69F0AE)',
                        }
                      }}
                    />
                    <Typography 
                      variant="body2" 
                      sx={{ mt: 1, textAlign: 'center', color: 'primary.light' }}
                    >
                      Uploading your file...
                    </Typography>
                  </Box>
                )}
                
                {localUploadResponse && !localUploadResponse.error && (
                  <Alert 
                    severity="success" 
                    sx={{ 
                      mt: 3,
                      border: '1px solid rgba(0, 230, 118, 0.3)',
                      '& .MuiAlert-icon': {
                        color: '#00E676',
                      }
                    }}
                    className="pulse"
                  >
                    {localUploadResponse.message || 'File uploaded successfully!'}
                  </Alert>
                )}
                
                {localUploadResponse && localUploadResponse.error && (
                  <Alert 
                    severity="error" 
                    sx={{ 
                      mt: 3,
                      border: '1px solid rgba(255, 82, 82, 0.3)'
                    }}
                  >
                    {localUploadResponse.error}
                  </Alert>
                )}
              </Paper>
            </motion.div>
          </Grid>
          
          {filePreview && (
            <Grid item xs={12}>
              <motion.div variants={itemVariants}>
                <Paper 
                  sx={{ 
                    p: 3, 
                    mb: 3,
                    border: '1px solid rgba(0, 230, 118, 0.2)',
                    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4)',
                  }}
                >
                  <Typography 
                    variant="h5" 
                    component="h2" 
                    gutterBottom
                    className="cyber-header"
                    sx={{ mb: 3 }}
                  >
                    Pandas Query Interface
                  </Typography>
                  
                  <QueryInterface filePreview={filePreview} />
                </Paper>
              </motion.div>
            </Grid>
          )}
        </Grid>
      </Paper>
    </motion.div>
  );
};

export default DataExplorer; 