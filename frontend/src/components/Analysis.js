import React, { useState, useEffect, useContext } from 'react';
import { 
  Paper, 
  Typography, 
  Box, 
  Grid, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  CircularProgress, 
  Alert, 
  Divider, 
  Card,
  CardContent,
  Button,
  Tab,
  Tabs
} from '@mui/material';
import { motion } from 'framer-motion';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
  LineChart,
  Line
} from 'recharts';
import api from '../services/api';
import { CSVContext } from '../App';

const Analysis = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [overview, setOverview] = useState(null);
  const [currentChart, setCurrentChart] = useState('overview');
  const [selectedColumn, setSelectedColumn] = useState('');
  const [chartData, setChartData] = useState(null);
  const [xColumn, setXColumn] = useState('');
  const [yColumn, setYColumn] = useState('');
  const [activeTab, setActiveTab] = useState(0);

  // COLORS for charts
  const COLORS = [
    '#00E676', '#1DE9B6', '#00B0FF', '#00E5FF', '#651FFF', 
    '#D500F9', '#FF1744', '#FF9100', '#FFEA00', '#76FF03'
  ];

  // Access CSV context
  const { isCSVUploaded } = useContext(CSVContext);

  useEffect(() => {
    // Fetch overview data on component mount
    fetchOverviewData();
  }, []);

  // Fetch overview data
  const fetchOverviewData = async () => {
    setLoading(true);
    setError(null);
    try {
      // Try to use real data
      const endpoint = '/stats_query?chart_type=overview';
      
      console.log(`Fetching overview data from: ${endpoint}`);
      const response = await api.get(endpoint);
      console.log("Overview data response:", response.data);
      
      setOverview(response.data);
      // Set default column if available
      if (response.data.numericColumns && response.data.numericColumns.length > 0) {
        setSelectedColumn(response.data.numericColumns[0]);
        setXColumn(response.data.numericColumns[0]);
        if (response.data.numericColumns.length > 1) {
          setYColumn(response.data.numericColumns[1]);
        } else {
          setYColumn(response.data.numericColumns[0]);
        }
      }
    } catch (error) {
      console.error("Error fetching overview data:", error);
      setError(`Failed to fetch overview data: ${error.message || "Unknown error"}`);
      setOverview(null);
    } finally {
      setLoading(false);
    }
  };
  
  // Fetch chart data
  const fetchChartData = async (chartType, params = {}) => {
    setLoading(true);
    setError(null);
    try {
      let queryParams = new URLSearchParams({
        chart_type: chartType,
      });
      
      // Add additional params
      Object.keys(params).forEach(key => {
        queryParams.append(key, params[key]);
      });
      
      // Use real data
      const endpoint = `/stats_query?${queryParams}`;
      
      console.log(`Fetching chart data from: ${endpoint}`);
      const response = await api.get(endpoint);
      console.log(`${chartType} data response:`, response.data);
      
      setChartData(response.data);
    } catch (error) {
      console.error(`Error fetching ${chartType} data:`, error);
      setError(`Failed to fetch ${chartType} data: ${error.message || "Unknown error"}`);
      setChartData(null);
    } finally {
      setLoading(false);
    }
  };

  const handleChartTypeChange = (event) => {
    const newChartType = event.target.value;
    setCurrentChart(newChartType);
    setChartData(null);
    
    if (newChartType === 'overview') {
      fetchOverviewData();
    } else if (newChartType === 'histogram' && selectedColumn) {
      fetchChartData('histogram', { column: selectedColumn });
    } else if (newChartType === 'correlation') {
      fetchChartData('correlation');
    } else if (newChartType === 'scatter' && xColumn && yColumn) {
      fetchChartData('scatter', { x_column: xColumn, y_column: yColumn });
    }
  };

  const handleColumnChange = (event) => {
    const newColumn = event.target.value;
    setSelectedColumn(newColumn);
    
    if (currentChart === 'histogram') {
      fetchChartData('histogram', { column: newColumn });
    }
  };

  const handleXColumnChange = (event) => {
    const newXColumn = event.target.value;
    setXColumn(newXColumn);
    
    if (currentChart === 'scatter' && yColumn) {
      fetchChartData('scatter', { x_column: newXColumn, y_column: yColumn });
    }
  };

  const handleYColumnChange = (event) => {
    const newYColumn = event.target.value;
    setYColumn(newYColumn);
    
    if (currentChart === 'scatter' && xColumn) {
      fetchChartData('scatter', { x_column: xColumn, y_column: newYColumn });
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
    
    switch(newValue) {
      case 0:
        setCurrentChart('overview');
        fetchOverviewData();
        break;
      case 1:
        setCurrentChart('histogram');
        if (selectedColumn) {
          fetchChartData('histogram', { column: selectedColumn });
        }
        break;
      case 2:
        setCurrentChart('correlation');
        fetchChartData('correlation');
        break;
      case 3:
        setCurrentChart('scatter');
        if (xColumn && yColumn) {
          fetchChartData('scatter', { x_column: xColumn, y_column: yColumn });
        }
        break;
      default:
        break;
    }
  };

  // Render the Overview section
  const renderOverview = () => {
    if (!overview) {
      return (
        <Box sx={{ p: 4, textAlign: 'center' }}>
          <Alert severity="info" sx={{ mb: 2 }}>
            No data found. Please upload a CSV file or generate synthetic data first.
          </Alert>
          <Button 
            variant="contained" 
            component="a" 
            href="/explorer" 
            sx={{ mr: 2 }}
          >
            Upload CSV File
          </Button>
          <Button 
            variant="outlined" 
            component="a" 
            href="/generation"
          >
            Generate Data
          </Button>
        </Box>
      );
    }

    return (
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card sx={{ 
            height: '100%',
            border: '1px solid rgba(0, 230, 118, 0.2)',
            bgcolor: 'rgba(0, 0, 0, 0.2)',
            transition: 'all 0.3s ease',
            '&:hover': {
              boxShadow: '0 0 15px rgba(0, 230, 118, 0.3)',
            }
          }}>
            <CardContent>
              <Typography variant="h6" sx={{ color: '#00E676', mb: 2 }}>
                Dataset Overview
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Box sx={{ 
                    p: 2, 
                    bgcolor: 'rgba(0, 230, 118, 0.05)', 
                    borderRadius: 1,
                    border: '1px solid rgba(0, 230, 118, 0.2)'
                  }}>
                    <Typography variant="h4" color="primary">{overview.rowCount}</Typography>
                    <Typography variant="body2">Rows</Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box sx={{ 
                    p: 2, 
                    bgcolor: 'rgba(0, 230, 118, 0.05)', 
                    borderRadius: 1,
                    border: '1px solid rgba(0, 230, 118, 0.2)'
                  }}>
                    <Typography variant="h4" color="primary">{overview.columnCount}</Typography>
                    <Typography variant="body2">Columns</Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box sx={{ 
                    p: 2, 
                    bgcolor: 'rgba(0, 230, 118, 0.05)', 
                    borderRadius: 1,
                    border: '1px solid rgba(0, 230, 118, 0.2)'
                  }}>
                    <Typography variant="h4" color="primary">{overview.numericColumnCount}</Typography>
                    <Typography variant="body2">Numeric Columns</Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box sx={{ 
                    p: 2, 
                    bgcolor: 'rgba(0, 230, 118, 0.05)', 
                    borderRadius: 1,
                    border: '1px solid rgba(0, 230, 118, 0.2)'
                  }}>
                    <Typography variant="h4" color="primary">{overview.categoricalColumnCount}</Typography>
                    <Typography variant="body2">Categorical Columns</Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card sx={{ 
            height: '100%',
            border: '1px solid rgba(0, 230, 118, 0.2)',
            bgcolor: 'rgba(0, 0, 0, 0.2)',
            transition: 'all 0.3s ease',
            '&:hover': {
              boxShadow: '0 0 15px rgba(0, 230, 118, 0.3)',
            }
          }}>
            <CardContent>
              <Typography variant="h6" sx={{ color: '#00E676', mb: 2 }}>
                Column Distribution
              </Typography>
              <Box sx={{ height: 180 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Numeric', value: overview.numericColumnCount },
                        { name: 'Categorical', value: overview.categoricalColumnCount }
                      ]}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      outerRadius={60}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {[
                        { name: 'Numeric', value: overview.numericColumnCount },
                        { name: 'Categorical', value: overview.categoricalColumnCount }
                      ].map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => [value, 'Count']} />
                  </PieChart>
                </ResponsiveContainer>
              </Box>
              <Divider sx={{ my: 2, borderColor: 'rgba(0, 230, 118, 0.2)' }} />
              <Typography variant="subtitle2" sx={{ color: '#00E676', mb: 1 }}>
                Missing Values
              </Typography>
              <Typography variant="h5" color="primary">
                {overview.missingValues}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  // Render the Histogram chart
  const renderHistogram = () => {
    if (!selectedColumn || !overview || !overview.numericColumns) {
      return (
        <Box sx={{ p: 4, textAlign: 'center' }}>
          <Alert severity="info">
            No data found. Please upload a CSV file or generate synthetic data first.
          </Alert>
        </Box>
      );
    }
    
    const allColumns = [...(overview.numericColumns || []), ...(overview.categoricalColumns || [])];
    
    return (
      <>
        <Box sx={{ mb: 3 }}>
          <FormControl fullWidth variant="outlined" sx={{ mb: 2 }}>
            <InputLabel>Select Column</InputLabel>
            <Select
              value={selectedColumn}
              onChange={handleColumnChange}
              label="Select Column"
              sx={{
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(0, 230, 118, 0.3)',
                },
                '&:hover .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(0, 230, 118, 0.5)',
                },
                '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                  borderColor: '#00E676',
                },
              }}
            >
              {allColumns.map((column) => (
                <MenuItem key={column} value={column}>{column}</MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <Button 
            variant="contained" 
            color="primary" 
            onClick={() => fetchChartData('histogram', { column: selectedColumn })}
            sx={{ mb: 2 }}
            className="glow-effect"
          >
            Generate Histogram
          </Button>
        </Box>
        
        {chartData && chartData.type === 'numeric' && (
          <Box sx={{ height: 400 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData.data}
                margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="bin" 
                  angle={-45} 
                  textAnchor="end" 
                  height={80} 
                  tick={{ fill: '#ccc', fontSize: 12 }} 
                />
                <YAxis tick={{ fill: '#ccc' }} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1a1a1a', 
                    borderColor: '#00E676',
                    color: '#fff'
                  }} 
                />
                <Legend />
                <Bar 
                  dataKey="count" 
                  name="Frequency" 
                  fill="#00E676" 
                  fillOpacity={0.8}
                  stroke="#00E676"
                  strokeWidth={1}
                />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        )}
        
        {chartData && chartData.type === 'categorical' && (
          <Box sx={{ height: 400 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData.data}
                margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
                layout="vertical"
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis type="number" tick={{ fill: '#ccc' }} />
                <YAxis 
                  dataKey="value" 
                  type="category" 
                  width={150}
                  tick={{ fill: '#ccc', fontSize: 12 }} 
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1a1a1a', 
                    borderColor: '#00E676',
                    color: '#fff'
                  }} 
                />
                <Legend />
                <Bar 
                  dataKey="count" 
                  name="Frequency" 
                  fill="#00E676" 
                  fillOpacity={0.8}
                  stroke="#00E676"
                  strokeWidth={1}
                />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        )}
      </>
    );
  };

  // Render the Correlation chart
  const renderCorrelation = () => {
    if (!chartData || !chartData.correlationMatrix) {
      return (
        <Box sx={{ p: 4, textAlign: 'center' }}>
          <Alert severity="info">
            No correlation data found. Please upload a CSV file or generate synthetic data with multiple numeric columns.
          </Alert>
        </Box>
      );
    }
    
    // Transform correlation data for visualization
    const transformedData = chartData.correlationMatrix.map(item => ({
      source: item.source,
      target: item.target,
      value: item.correlation,
      // Color based on correlation strength
      fill: item.correlation > 0.7 ? '#00E676' :
            item.correlation > 0.5 ? '#76FF03' :
            item.correlation > 0.3 ? '#FFEA00' :
            item.correlation > 0 ? '#FF9100' :
            item.correlation > -0.3 ? '#FF1744' :
            item.correlation > -0.5 ? '#D500F9' : '#651FFF'
    }));
    
    // Get top 5 strongest correlations
    const topCorrelations = [...transformedData]
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 5);
    
    return (
      <>
        <Typography variant="h6" sx={{ color: '#00E676', mb: 2 }}>
          Top Correlations
        </Typography>
        
        <Grid container spacing={2} sx={{ mb: 3 }}>
          {topCorrelations.map((item, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Box sx={{ 
                p: 2, 
                borderRadius: 1,
                bgcolor: 'rgba(0, 0, 0, 0.2)',
                border: '1px solid rgba(0, 230, 118, 0.2)',
                '&:hover': {
                  boxShadow: '0 0 10px rgba(0, 230, 118, 0.3)',
                },
              }}>
                <Typography variant="subtitle2" gutterBottom>
                  {item.source} ↔ {item.target}
                </Typography>
                <Typography 
                  variant="h4" 
                  sx={{ 
                    color: item.fill, 
                    fontWeight: 'bold',
                    textAlign: 'center' 
                  }}
                >
                  {item.value.toFixed(2)}
                </Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
        
        <Box sx={{ height: 400, mt: 4 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart
              margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis 
                type="category" 
                dataKey="source" 
                name="Source" 
                tick={{ fill: '#ccc' }} 
                allowDuplicatedCategory={false}
              />
              <YAxis 
                type="category" 
                dataKey="target" 
                name="Target" 
                tick={{ fill: '#ccc' }} 
                allowDuplicatedCategory={false}
              />
              <ZAxis 
                dataKey="value" 
                range={[20, 200]} 
                name="Correlation" 
              />
              <Tooltip 
                formatter={(value) => [value.toFixed(4), 'Correlation']}
                contentStyle={{ 
                  backgroundColor: '#1a1a1a', 
                  borderColor: '#00E676',
                  color: '#fff'
                }}
              />
              <Scatter 
                data={transformedData} 
                fill="#00E676"
              >
                {transformedData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </Box>
      </>
    );
  };

  // Render the Scatter Plot
  const renderScatterPlot = () => {
    if (!xColumn || !yColumn || !chartData || !chartData.scatterData) {
      return (
        <Box sx={{ p: 4, textAlign: 'center' }}>
          <Alert severity="info">
            No scatter plot data found. Please upload a CSV file or generate synthetic data with multiple numeric columns.
          </Alert>
        </Box>
      );
    }
    
    return (
      <>
        <Box sx={{ mb: 3 }}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={5}>
              <FormControl fullWidth variant="outlined">
                <InputLabel>X-Axis Column</InputLabel>
                <Select
                  value={xColumn}
                  onChange={handleXColumnChange}
                  label="X-Axis Column"
                  sx={{
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'rgba(0, 230, 118, 0.3)',
                    },
                    '&:hover .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'rgba(0, 230, 118, 0.5)',
                    },
                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                      borderColor: '#00E676',
                    },
                  }}
                >
                  {overview.numericColumns.map((column) => (
                    <MenuItem key={column} value={column}>{column}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={5}>
              <FormControl fullWidth variant="outlined">
                <InputLabel>Y-Axis Column</InputLabel>
                <Select
                  value={yColumn}
                  onChange={handleYColumnChange}
                  label="Y-Axis Column"
                  sx={{
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'rgba(0, 230, 118, 0.3)',
                    },
                    '&:hover .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'rgba(0, 230, 118, 0.5)',
                    },
                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                      borderColor: '#00E676',
                    },
                  }}
                >
                  {overview.numericColumns.map((column) => (
                    <MenuItem key={column} value={column}>{column}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={2}>
              <Button 
                variant="contained" 
                color="primary" 
                fullWidth
                sx={{ height: '56px' }}
                onClick={() => fetchChartData('scatter', { x_column: xColumn, y_column: yColumn })}
                className="glow-effect"
              >
                Plot
              </Button>
            </Grid>
          </Grid>
        </Box>
        
        {chartData && chartData.scatterData && (
          <Box sx={{ height: 400 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart
                margin={{ top: 20, right: 20, bottom: 60, left: 30 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  type="number" 
                  dataKey="x" 
                  name={chartData.x_column} 
                  label={{ 
                    value: chartData.x_column, 
                    position: 'bottom', 
                    fill: '#ccc', 
                    offset: 0 
                  }}
                  tick={{ fill: '#ccc' }}
                />
                <YAxis 
                  type="number" 
                  dataKey="y" 
                  name={chartData.y_column}
                  label={{ 
                    value: chartData.y_column, 
                    angle: -90, 
                    position: 'left', 
                    fill: '#ccc',
                    offset: 10
                  }}
                  tick={{ fill: '#ccc' }}
                />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  formatter={(value, name) => [value, name === 'x' ? chartData.x_column : chartData.y_column]}
                  contentStyle={{ 
                    backgroundColor: '#1a1a1a', 
                    borderColor: '#00E676',
                    color: '#fff'
                  }}
                />
                <Scatter 
                  name="Data Points" 
                  data={chartData.scatterData} 
                  fill="#00E676"
                  opacity={0.7}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </Box>
        )}
      </>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
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
          Data Analysis
        </Typography>
        
        {error && (
          <Alert 
            severity="error" 
            sx={{ mb: 3, border: '1px solid rgba(255, 82, 82, 0.3)' }}
          >
            {error}
          </Alert>
        )}
        
        <Tabs 
          value={activeTab} 
          onChange={handleTabChange}
          variant="fullWidth"
          sx={{
            mb: 4,
            '& .MuiTabs-indicator': {
              backgroundColor: '#00E676',
            },
            '& .MuiTab-root': {
              color: 'rgba(255, 255, 255, 0.7)',
              '&.Mui-selected': {
                color: '#00E676',
              },
            },
          }}
        >
          <Tab label="Overview" />
          <Tab label="Histogram" />
          <Tab label="Correlation" />
          <Tab label="Scatter Plot" />
        </Tabs>
        
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress color="primary" />
          </Box>
        ) : (
          <Box>
            {activeTab === 0 && renderOverview()}
            {activeTab === 1 && renderHistogram()}
            {activeTab === 2 && renderCorrelation()}
            {activeTab === 3 && renderScatterPlot()}
          </Box>
        )}
      </Paper>
    </motion.div>
  );
};

export default Analysis; 