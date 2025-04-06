import React, { useState, useEffect, useContext } from 'react';
import { 
  Typography, 
  Box, 
  TextField, 
  Slider, 
  Button,
  Divider,
  Stack,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  LinearProgress,
  Alert,
  AlertTitle
} from '@mui/material';
import { motion } from 'framer-motion';
import api from '../services/api';
// Import context from App.js
import { CSVContext, GenerationContext, UploadResponseContext } from '../App';

const DataGeneration = () => {
  // State for generation parameters
  const [numSamples, setNumSamples] = useState(() => {
    const saved = localStorage.getItem('data_generation_numSamples');
    return saved ? parseInt(saved, 10) : 10;
  });
  const [temperature, setTemperature] = useState(() => {
    const saved = localStorage.getItem('data_generation_temperature');
    return saved ? parseFloat(saved) : 0.7;
  });
  const [topP, setTopP] = useState(() => {
    const saved = localStorage.getItem('data_generation_topP');
    return saved ? parseFloat(saved) : 0.9;
  });
  const [repetitionPenalty, setRepetitionPenalty] = useState(() => {
    const saved = localStorage.getItem('data_generation_repetitionPenalty');
    return saved ? parseFloat(saved) : 1.1;
  });
  const [maxTokens, setMaxTokens] = useState(() => {
    const saved = localStorage.getItem('data_generation_maxTokens');
    return saved ? parseInt(saved, 10) : 2048;
  });
  // Fixed model value
  const model = 'gpt-4o-mini';
  
  // State for data generation and display
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [generatedData, setGeneratedData] = useState(null);
  const [columns, setColumns] = useState([]);
  const [error, setError] = useState(null);
  const [totalRows, setTotalRows] = useState(0);
  const [originalCount, setOriginalCount] = useState(0);
  const [syntheticCount, setSyntheticCount] = useState(0);
  const [isCombined, setIsCombined] = useState(false);
  const [showCombined, setShowCombined] = useState(true);
  
  // Access the contexts
  const { isCSVUploaded, setCSVUploaded } = useContext(CSVContext);
  const { setGenerationStatus } = useContext(GenerationContext);
  const { setUploadResponse } = useContext(UploadResponseContext);
  
  // Check for ongoing generation on component mount
  useEffect(() => {
    // Check if there's an ongoing generation when component mounts
    const storedGenerationStatus = localStorage.getItem('generationStatus');
    if (storedGenerationStatus) {
      try {
        const status = JSON.parse(storedGenerationStatus);
        // Only restore if generation is still in progress
        if (status.isGenerating) {
          setIsGenerating(true);
          setGenerationProgress(status.progress || 0);
          // Update global state
          setGenerationStatus({
            isGenerating: true,
            progress: status.progress || 0,
            currentFile: null,
            error: null
          });
          checkGenerationStatus(); // Immediately check current status
        } else {
          // Clear localStorage if generation is complete
          localStorage.removeItem('generationStatus');
        }
      } catch (e) {
        console.error("Error parsing stored generation status", e);
        localStorage.removeItem('generationStatus');
      }
    }
    
    // Check for previously generated data
    try {
      const savedData = localStorage.getItem('generatedData');
      const savedColumns = localStorage.getItem('generatedColumns');
      const savedMetadata = localStorage.getItem('generatedMetadata');
      
      if (savedData && savedColumns && savedMetadata) {
        const parsedData = JSON.parse(savedData);
        const parsedColumns = JSON.parse(savedColumns);
        const parsedMetadata = JSON.parse(savedMetadata);
        
        setGeneratedData(parsedData);
        setColumns(parsedColumns);
        setTotalRows(parsedMetadata.totalRows || parsedData.length);
        setOriginalCount(parsedMetadata.originalCount || 0);
        setSyntheticCount(parsedMetadata.syntheticCount || 0);
        setIsCombined(parsedMetadata.isCombined || false);
        console.log("Restored previously generated data from localStorage");
      }
    } catch (e) {
      console.error("Error restoring generated data:", e);
    }
  }, [setGenerationStatus]);
  
  // Save parameters to localStorage when they change
  useEffect(() => {
    localStorage.setItem('data_generation_numSamples', numSamples.toString());
  }, [numSamples]);
  
  useEffect(() => {
    localStorage.setItem('data_generation_temperature', temperature.toString());
  }, [temperature]);
  
  useEffect(() => {
    localStorage.setItem('data_generation_topP', topP.toString());
  }, [topP]);
  
  useEffect(() => {
    localStorage.setItem('data_generation_repetitionPenalty', repetitionPenalty.toString());
  }, [repetitionPenalty]);
  
  useEffect(() => {
    localStorage.setItem('data_generation_maxTokens', maxTokens.toString());
  }, [maxTokens]);
  
  // Effect to fetch data when showCombined changes
  useEffect(() => {
    if (generatedData) {
      fetchGeneratedData();
    }
  }, [showCombined]);
  
  // Poll for generation status when isGenerating is true
  useEffect(() => {
    let interval;
    
    if (isGenerating) {
      interval = setInterval(checkGenerationStatus, 1000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isGenerating]);
  
  // Update global state when local state changes
  useEffect(() => {
    setGenerationStatus({
      isGenerating,
      progress: generationProgress,
      currentFile: null,
      error: error
    });
  }, [isGenerating, generationProgress, error, setGenerationStatus]);
  
  // Check generation status
  const checkGenerationStatus = async () => {
    try {
      const response = await api.getGenerationStatus();
      
      // Only update if progress has changed
      if (response.progress !== generationProgress) {
        setGenerationProgress(response.progress || 0);
        
        // Store generation status in localStorage for persistence across tab switches
        localStorage.setItem('generationStatus', JSON.stringify({
          isGenerating: response.isGenerating,
          progress: response.progress || 0,
          timestamp: new Date().getTime()
        }));
        
        // Log progress to console for debugging
        console.log(`Generation progress update: ${response.progress.toFixed(1)}%`);
      }
      
      // If generation is complete, fetch the results
      if (!response.isGenerating && response.progress >= 100) {
        setGenerationProgress(100);
        setIsGenerating(false);
        localStorage.removeItem('generationStatus'); // Clear localStorage
        
        // Only fetch data if we have generated data and no error
        if (response.has_generated_data) {
          console.log("Generation complete, fetching data...");
          fetchGeneratedData();
        } else if (response.error) {
          setError(response.error_details || response.error);
        }
      } else if (!response.isGenerating && response.progress < 100) {
        // Handle case where generation stopped before completion
        setIsGenerating(false);
        localStorage.removeItem('generationStatus'); // Clear localStorage
        if (response.error) {
          setError(response.error_details || response.error);
        } else if (response.has_generated_data) {
          console.log("Generation stopped but data available, fetching data...");
          fetchGeneratedData();
        } else {
          setError('Data generation was interrupted before completion');
        }
      }
    } catch (error) {
      console.error("Error checking generation status:", error);
      setError('Failed to check generation status: ' + (error.message || 'Unknown error'));
      setIsGenerating(false);
      localStorage.removeItem('generationStatus'); // Clear localStorage on error
    }
  };
  
  // Fetch generated data
  const fetchGeneratedData = async () => {
    try {
      const response = await api.get(`/get_generated_data${showCombined ? '' : '?combined=false'}`);
      
      if (response.data.success) {
        // Save the data to state
        setGeneratedData(response.data.data);
        setColumns(response.data.columns);
        setTotalRows(response.data.rowCount || response.data.data.length);
        setError(null);
        setOriginalCount(response.data.originalCount || 0);
        setSyntheticCount(response.data.syntheticCount || 0);
        setIsCombined(response.data.isCombined || false);
        
        // Store the data in localStorage for persistence
        try {
          localStorage.setItem('generatedData', JSON.stringify(response.data.data));
          localStorage.setItem('generatedColumns', JSON.stringify(response.data.columns));
          localStorage.setItem('generatedMetadata', JSON.stringify({
            totalRows: response.data.rowCount || response.data.data.length,
            originalCount: response.data.originalCount || 0,
            syntheticCount: response.data.syntheticCount || 0,
            isCombined: response.data.isCombined || false
          }));
        } catch (storageError) {
          // If storing fails (e.g., due to size limits), just log the error
          console.error("Error storing generated data in localStorage:", storageError);
        }
        
        // Update the CSV status globally after successful generation
        try {
          const csvStatus = await api.checkCSVStatus();
          if (csvStatus && csvStatus.has_csv) {
            // Update the upload response using the context
            setUploadResponse(csvStatus);
            
            // Ensure CSV uploaded flag is set
            setCSVUploaded(true);
          }
        } catch (statusError) {
          console.error("Error updating CSV status after generation:", statusError);
        }
      } else {
        setError(response.data.error || 'Failed to fetch generated data');
      }
    } catch (error) {
      console.error("Error fetching generated data:", error);
      setError(error.message || 'Failed to fetch generated data');
    }
  };

  // Handle form submission
  const handleGenerate = async () => {
    setError(null);
    setIsGenerating(true);
    setGenerationProgress(0);
    
    // Clear any previously stored data since we're generating new data
    setGeneratedData(null); 
    localStorage.removeItem('generatedData');
    localStorage.removeItem('generatedColumns');
    localStorage.removeItem('generatedMetadata');
    
    // Store initial generation status in localStorage
    localStorage.setItem('generationStatus', JSON.stringify({
      isGenerating: true,
      progress: 0,
      timestamp: new Date().getTime()
    }));
    
    try {
      // Send parameters to the backend
      const response = await api.post('/generate_data', {
        numSamples,
        temperature,
        topP,
        repetitionPenalty,
        maxTokens,
        model
      });
      
      if (!response.data.success) {
        setError(response.data.error || 'Failed to start data generation');
        setIsGenerating(false);
        localStorage.removeItem('generationStatus'); // Clear localStorage on error
      }
    } catch (error) {
      console.error("Error starting data generation:", error);
      setError(error.message || 'Failed to start data generation');
      setIsGenerating(false);
      localStorage.removeItem('generationStatus'); // Clear localStorage on error
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      style={{ 
        width: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        minHeight: '100%',
      }}
    >
      {/* Page Header */}
      <Typography 
        variant="h4" 
        component="h1" 
        gutterBottom
        className="cyber-header"
        sx={{ mb: 3 }}
      >
        Data Generation
      </Typography>
      
      {/* Main Content Area */}
      <Box sx={{ 
        display: 'flex', 
        flexGrow: 1, 
        gap: 3, 
        width: '100%',
        minHeight: 'calc(100vh - 200px)',
      }}>
        {/* Main Card - Data Preview/Results Area */}
        <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
          <Card sx={{ 
            flexGrow: 1,
            display: 'flex',
            flexDirection: 'column',
            border: '1px solid rgba(0, 230, 118, 0.2)',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4)',
            bgcolor: 'rgba(0, 0, 0, 0.2)',
            overflow: 'hidden'
          }}>
            <CardContent sx={{ 
              p: 3, 
              display: 'flex', 
              flexDirection: 'column',
              flexGrow: 1,
              overflow: 'hidden'
            }}>
              <Typography variant="h6" sx={{ color: '#00E676', mb: 2 }}>
                Data Preview & Generation Results
              </Typography>
              
              {/* Persistent Progress Bar */}
              {isGenerating && (
                <Box sx={{ width: '100%', mb: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="subtitle2" sx={{ color: 'rgba(0, 230, 118, 0.9)' }}>
                      Generating {numSamples} samples...
                    </Typography>
                    <Typography variant="subtitle2" sx={{ color: 'rgba(0, 230, 118, 0.9)' }}>
                      {Math.round(generationProgress)}%
                    </Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={generationProgress} 
                    sx={{ 
                      height: 8, 
                      borderRadius: 2,
                      backgroundColor: 'rgba(0, 0, 0, 0.3)',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: 'rgba(0, 230, 118, 0.8)',
                        borderRadius: 2,
                      }
                    }}
                  />
                </Box>
              )}
              
              {/* Generation Status and Error Messaging */}
              {isGenerating && (
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <CircularProgress size={24} sx={{ color: '#00E676', mr: 2 }} />
                  <Typography>
                    Processing... this may take a few minutes for larger datasets
                  </Typography>
                </Box>
              )}
              
              {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  <AlertTitle>Error</AlertTitle>
                  {error}
                </Alert>
              )}
              
              {/* Data Table Display */}
              <Box 
                sx={{ 
                  flexGrow: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: generatedData ? 'flex-start' : 'center',
                  alignItems: generatedData ? 'stretch' : 'center',
                  overflow: 'hidden'
                }}
              >
                {!generatedData && !isGenerating ? (
                  <Typography variant="body1" color="text.secondary">
                    Configure parameters and click Generate Data to create synthetic data.
                  </Typography>
                ) : generatedData ? (
                  <>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                      Showing {generatedData.length} of {totalRows} rows
                      {isCombined && (
                        <span style={{ marginLeft: '8px', color: 'rgba(0, 230, 118, 0.8)' }}>
                          ({originalCount} original + {syntheticCount} synthetic)
                        </span>
                      )}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1, justifyContent: 'space-between' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Box sx={{ 
                          display: 'inline-block', 
                          width: '12px', 
                          height: '12px', 
                          bgcolor: 'rgba(0, 0, 0, 0.3)', 
                          mr: 0.5,
                          border: '1px solid rgba(255, 255, 255, 0.2)'
                        }} />
                        <Typography variant="caption" sx={{ mr: 2 }}>Original Data</Typography>
                        
                        <Box sx={{ 
                          display: 'inline-block', 
                          width: '12px', 
                          height: '12px', 
                          bgcolor: 'rgba(0, 230, 118, 0.15)', 
                          mr: 0.5,
                          border: '1px solid rgba(0, 230, 118, 0.3)'
                        }} />
                        <Typography variant="caption">Synthetic Data</Typography>
                      </Box>
                      
                      {isCombined && (
                        <Button 
                          size="small" 
                          variant="outlined"
                          onClick={() => setShowCombined(!showCombined)}
                          sx={{ 
                            fontSize: '0.7rem', 
                            py: 0.5,
                            color: 'rgba(0, 230, 118, 0.8)',
                            borderColor: 'rgba(0, 230, 118, 0.3)',
                            '&:hover': {
                              borderColor: 'rgba(0, 230, 118, 0.8)',
                              bgcolor: 'rgba(0, 230, 118, 0.05)'
                            }
                          }}
                        >
                          {showCombined ? 'Show Only Synthetic' : 'Show All Data'}
                        </Button>
                      )}
                    </Box>
                    <Paper 
                      sx={{ 
                        flexGrow: 1, 
                        display: 'flex',
                        flexDirection: 'column',
                        overflow: 'hidden',
                        bgcolor: 'rgba(0, 0, 0, 0.2)',
                        border: '1px solid rgba(0, 230, 118, 0.1)',
                        minHeight: '400px'
                      }}
                    >
                      <TableContainer sx={{ 
                        flexGrow: 1,
                        height: '100%',
                        overflow: 'auto'
                      }}>
                        <Table stickyHeader size="small">
                          <TableHead>
                            <TableRow>
                              {columns.map((column) => (
                                <TableCell 
                                  key={column}
                                  sx={{ 
                                    bgcolor: 'rgba(0, 0, 0, 0.7)',
                                    color: '#00E676',
                                    fontWeight: 'bold'
                                  }}
                                >
                                  {column}
                                </TableCell>
                              ))}
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {generatedData.map((row, rowIndex) => (
                              <TableRow 
                                key={rowIndex}
                                sx={{ 
                                  '&:nth-of-type(odd)': { 
                                    bgcolor: row.is_synthetic 
                                      ? 'rgba(0, 230, 118, 0.15)' 
                                      : 'rgba(0, 0, 0, 0.3)' 
                                  },
                                  '&:nth-of-type(even)': { 
                                    bgcolor: row.is_synthetic 
                                      ? 'rgba(0, 230, 118, 0.1)' 
                                      : 'rgba(0, 0, 0, 0.2)' 
                                  },
                                  '&:hover': { 
                                    bgcolor: row.is_synthetic 
                                      ? 'rgba(0, 230, 118, 0.25)' 
                                      : 'rgba(0, 230, 118, 0.1)'
                                  },
                                  borderLeft: row.is_synthetic 
                                    ? '2px solid rgba(0, 230, 118, 0.5)' 
                                    : 'none'
                                }}
                              >
                                {columns.map((column) => (
                                  <TableCell 
                                    key={`${rowIndex}-${column}`}
                                    sx={{ 
                                      color: 'white',
                                      borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
                                    }}
                                  >
                                    {row[column]}
                                  </TableCell>
                                ))}
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Paper>
                  </>
                ) : null}
              </Box>
            </CardContent>
          </Card>
        </Box>
        
        {/* Right Side Panel - Parameters */}
        <Box 
          sx={{ 
            width: '280px',
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            bgcolor: 'rgba(0, 0, 0, 0.2)',
            borderRadius: '8px',
            p: 2,
            border: '1px solid rgba(0, 230, 118, 0.1)'
          }}
        >
          <Typography variant="h6" sx={{ color: '#00E676', mb: 2, fontSize: '1rem' }}>
            Generation Parameters (Model: GPT-4o Mini)
          </Typography>
          
          {/* Sidebar Progress Bar */}
          {isGenerating && (
            <Box sx={{ width: '100%', mb: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="caption" sx={{ color: 'rgba(0, 230, 118, 0.9)' }}>
                  Progress
                </Typography>
                <Typography variant="caption" sx={{ color: 'rgba(0, 230, 118, 0.9)' }}>
                  {Math.round(generationProgress)}%
                </Typography>
              </Box>
              <LinearProgress 
                variant="determinate" 
                value={generationProgress} 
                sx={{ 
                  height: 6, 
                  borderRadius: 1,
                  backgroundColor: 'rgba(0, 0, 0, 0.3)',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: 'rgba(0, 230, 118, 0.8)',
                    borderRadius: 1,
                  }
                }}
              />
              <Typography variant="caption" sx={{ display: 'block', mt: 1, textAlign: 'center', color: 'rgba(255, 255, 255, 0.6)' }}>
                Generating {numSamples} samples
              </Typography>
              <Divider sx={{ borderColor: 'rgba(0, 230, 118, 0.2)', my: 2 }} />
            </Box>
          )}
          
          <Stack spacing={1.5} sx={{ mb: 'auto' }}>
            {/* Number of Samples */}
            <TextField
              size="small"
              fullWidth
              label="Number of Samples"
              type="number"
              value={numSamples}
              onChange={(e) => {
                const value = Math.max(1, parseInt(e.target.value, 10) || 1);
                setNumSamples(value);
              }}
              sx={{ 
                '& .MuiOutlinedInput-root': {
                  '& fieldset': { borderColor: 'rgba(0, 230, 118, 0.3)' },
                  '&:hover fieldset': { borderColor: 'rgba(0, 230, 118, 0.5)' },
                  '&.Mui-focused fieldset': { borderColor: '#00E676' },
                },
                '& .MuiInputLabel-root': { color: 'rgba(255, 255, 255, 0.7)' },
                '& .MuiInputBase-input': { color: 'white' },
              }}
              InputProps={{
                inputProps: { min: 1, max: 10000 }
              }}
              disabled={isGenerating}
            />
            
            <Divider sx={{ borderColor: 'rgba(0, 230, 118, 0.2)' }} />
            
            {/* Temperature */}
            <Box>
              <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)', display: 'flex', justifyContent: 'space-between' }}>
                <span>Temperature</span>
                <span>{temperature}</span>
              </Typography>
              <Slider
                size="small"
                value={temperature}
                onChange={(e, newValue) => {
                  setTemperature(parseFloat(newValue.toFixed(1)));
                }}
                min={0.1}
                max={2.0}
                step={0.1}
                sx={{ color: '#00E676', py: 0, mt: 0.5 }}
                disabled={isGenerating}
              />
            </Box>
            
            {/* Top-P */}
            <Box>
              <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)', display: 'flex', justifyContent: 'space-between' }}>
                <span>Top-P</span>
                <span>{topP}</span>
              </Typography>
              <Slider
                size="small"
                value={topP}
                onChange={(e, newValue) => {
                  setTopP(parseFloat(newValue.toFixed(2)));
                }}
                min={0.1}
                max={1.0}
                step={0.05}
                sx={{ color: '#00E676', py: 0, mt: 0.5 }}
                disabled={isGenerating}
              />
            </Box>
            
            {/* Frequency Penalty (renamed from Repetition Penalty) */}
            <Box>
              <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)', display: 'flex', justifyContent: 'space-between' }}>
                <span>Frequency Penalty</span>
                <span>{repetitionPenalty}</span>
              </Typography>
              <Slider
                size="small"
                value={repetitionPenalty}
                onChange={(e, newValue) => {
                  setRepetitionPenalty(parseFloat(newValue.toFixed(1)));
                }}
                min={1.0}
                max={2.0}
                step={0.1}
                sx={{ color: '#00E676', py: 0, mt: 0.5 }}
                disabled={isGenerating}
              />
            </Box>
            
            {/* Max Tokens */}
            <Box>
              <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)', display: 'flex', justifyContent: 'space-between' }}>
                <span>Max Tokens</span>
                <span>{maxTokens}</span>
              </Typography>
              <Slider
                size="small"
                value={maxTokens}
                onChange={(e, newValue) => {
                  setMaxTokens(parseInt(newValue, 10));
                }}
                min={256}
                max={4096}
                step={256}
                sx={{ color: '#00E676', py: 0, mt: 0.5 }}
                disabled={isGenerating}
              />
            </Box>
          </Stack>
          
          {/* Generate Button */}
          <Button
            variant="contained"
            color="primary"
            fullWidth
            onClick={handleGenerate}
            disabled={isGenerating}
            sx={{ 
              mt: 2,
              borderRadius: '4px',
              background: isGenerating ? 'rgba(0, 200, 83, 0.5)' : 'linear-gradient(90deg, #00C853, #00E676)',
              '&:hover': {
                background: isGenerating ? 'rgba(0, 200, 83, 0.5)' : 'linear-gradient(90deg, #00B34A, #00D26A)',
              },
              boxShadow: '0 2px 10px rgba(0, 230, 118, 0.3)'
            }}
            className="glow-effect"
            startIcon={isGenerating ? <CircularProgress size={16} sx={{ color: 'white' }} /> : null}
          >
            {isGenerating ? 'Generating...' : 'Generate Data'}
          </Button>
        </Box>
      </Box>
    </motion.div>
  );
};

export default DataGeneration; 