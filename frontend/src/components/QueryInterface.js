import React, { useState } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  TableContainer, 
  Table, 
  TableHead, 
  TableBody, 
  TableRow, 
  TableCell, 
  Paper, 
  Typography,
  Alert,
  CircularProgress,
  Chip,
  Divider
} from '@mui/material';
import { motion } from 'framer-motion';
import CodeIcon from '@mui/icons-material/Code';
import SearchIcon from '@mui/icons-material/Search';
import InfoIcon from '@mui/icons-material/Info';
import api from '../services/api';

const QueryInterface = ({ filePreview }) => {
  const [queryString, setQueryString] = useState('');
  const [queryResult, setQueryResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [examples, setExamples] = useState([
    { 
      label: 'Equal to', 
      query: `column_name == 'value'` 
    },
    { 
      label: 'Greater than', 
      query: 'numeric_column > 100' 
    },
    { 
      label: 'Contains text', 
      query: "text_column.str.contains('pattern')" 
    },
    { 
      label: 'AND condition', 
      query: "column1 > 0 and column2 == 'value'" 
    },
  ]);

  // Handle query input change
  const handleQueryChange = (event) => {
    setQueryString(event.target.value);
  };

  // Set query from example
  const setExampleQuery = (query) => {
    setQueryString(query);
  };

  // Handle query submission
  const handleSubmitQuery = async () => {
    if (!queryString.trim()) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await api.queryCSV(queryString);
      setQueryResult(result);
    } catch (error) {
      console.error('Error executing query:', error);
      setError(error.message || 'Failed to execute query');
      setQueryResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="subtitle1" sx={{ mb: 1 }}>
        Enter a pandas query to filter the data:
      </Typography>
      
      <TextField
        fullWidth
        variant="outlined"
        value={queryString}
        onChange={handleQueryChange}
        placeholder="e.g., Age > 30 and Department == 'Sales'"
        sx={{
          mb: 2,
          '& .MuiOutlinedInput-root': {
            '& fieldset': {
              borderColor: 'rgba(0, 230, 118, 0.3)',
            },
            '&:hover fieldset': {
              borderColor: 'rgba(0, 230, 118, 0.5)',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#00E676',
            },
          },
        }}
        InputProps={{
          startAdornment: <CodeIcon sx={{ mr: 1, color: 'rgba(0, 230, 118, 0.7)' }} />,
        }}
      />
      
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3, flexWrap: 'wrap', gap: 1 }}>
        <Button
          variant="contained"
          onClick={handleSubmitQuery}
          disabled={isLoading || !queryString.trim()}
          startIcon={<SearchIcon />}
          sx={{ 
            borderRadius: '8px',
            mb: { xs: 1, sm: 0 }
          }}
          className="glow-effect"
        >
          {isLoading ? 'Running Query...' : 'Run Query'}
        </Button>
        
        <Typography variant="body2" sx={{ ml: 2, mr: 1, color: 'text.secondary' }}>
          Examples:
        </Typography>
        
        {examples.map((example, index) => (
          <Chip
            key={index}
            label={example.label}
            clickable
            color="primary"
            variant="outlined"
            onClick={() => setExampleQuery(example.query)}
            sx={{ 
              borderColor: 'rgba(0, 230, 118, 0.4)',
              '&:hover': {
                borderColor: 'rgba(0, 230, 118, 0.7)',
                backgroundColor: 'rgba(0, 230, 118, 0.1)',
              }
            }}
          />
        ))}
      </Box>
      
      <Divider sx={{ my: 2, borderColor: 'rgba(0, 230, 118, 0.2)' }} />
      
      {isLoading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress sx={{ color: '#00E676' }} />
        </Box>
      )}
      
      {error && (
        <Alert 
          severity="error" 
          sx={{ 
            mb: 3,
            border: '1px solid rgba(255, 82, 82, 0.3)'
          }}
        >
          {error}
        </Alert>
      )}
      
      {!isLoading && !error && queryResult && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="h6" sx={{ color: '#00E676' }}>
              Query Results
            </Typography>
            <Box>
              <Chip 
                label={`${queryResult.matchedRowCount} rows matched`} 
                size="small" 
                sx={{ mr: 1, bgcolor: 'rgba(0, 230, 118, 0.1)' }}
              />
              {queryResult.truncated && (
                <Chip 
                  icon={<InfoIcon fontSize="small" />} 
                  label="Showing first 100 rows" 
                  size="small" 
                  sx={{ bgcolor: 'rgba(255, 193, 7, 0.1)', color: '#FFC107' }}
                />
              )}
            </Box>
          </Box>
          
          {queryResult.data && queryResult.data.length > 0 ? (
            <TableContainer 
              component={Paper} 
              sx={{ 
                maxHeight: 400, 
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
            >
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ bgcolor: 'rgba(0, 0, 0, 0.5)' }}>#</TableCell>
                    {queryResult.columns.map((col, index) => (
                      <TableCell key={index}>{col}</TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {queryResult.data.map((row, rowIndex) => (
                    <TableRow key={rowIndex} hover>
                      <TableCell sx={{ bgcolor: 'rgba(0, 0, 0, 0.3)' }}>{rowIndex + 1}</TableCell>
                      {queryResult.columns.map((col, colIndex) => (
                        <TableCell key={colIndex}>
                          {row[col] !== null && row[col] !== undefined ? String(row[col]) : ''}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Alert severity="info" sx={{ border: '1px solid rgba(33, 150, 243, 0.3)' }}>
              No data matched your query criteria.
            </Alert>
          )}
        </motion.div>
      )}
      
      <Box sx={{ mt: 3 }}>
        <Typography variant="subtitle2" sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
          <InfoIcon fontSize="small" sx={{ mr: 1, color: 'rgba(0, 230, 118, 0.7)' }} />
          Pandas Query Syntax
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Use pandas query syntax to filter the data. You can use:
        </Typography>
        <Box component="ul" sx={{ ml: 2, color: 'text.secondary' }}>
          <Typography component="li" variant="body2">
            Column comparisons: <code>column_name &gt; 100</code>, <code>column_name == 'value'</code>
          </Typography>
          <Typography component="li" variant="body2">
            Logical operators: <code>and</code>, <code>or</code>, <code>not</code>
          </Typography>
          <Typography component="li" variant="body2">
            String methods: <code>column_name.str.contains('pattern')</code>, <code>column_name.str.startswith('prefix')</code>
          </Typography>
          <Typography component="li" variant="body2">
            For column names with spaces, use backticks: <code>`First Name` == 'John'</code>
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default QueryInterface; 