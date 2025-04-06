import React, { useState } from 'react';
import { 
  Box, TextField, Button, CircularProgress, Paper, 
  Typography, Alert, useTheme, Divider
} from '@mui/material';
import { 
  ScatterChart, Scatter, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, BarChart, Bar, Cell
} from 'recharts';
import { motion } from 'framer-motion';
import api from '../services/api';

const EmbeddingVisualizer = () => {
  const theme = useTheme();
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [embeddingData, setEmbeddingData] = useState(null);
  const [similarityData, setSimilarityData] = useState(null);
  
  const handleQueryChange = (e) => {
    setQuery(e.target.value);
  };
  
  const handleVisualize = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.visualizeEmbeddings(query);
      if (response.success) {
        setEmbeddingData(response.data);
        setSimilarityData(response.similarities);
      } else {
        setError(response.error || "Failed to visualize embeddings");
      }
    } catch (error) {
      console.error("Error visualizing embeddings:", error);
      setError(error.message || "Failed to visualize embeddings");
    } finally {
      setLoading(false);
    }
  };
  
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <Paper sx={{ 
          p: 2, 
          maxWidth: 300, 
          bgcolor: 'rgba(0, 0, 0, 0.85)',
          border: data.type === 'query' 
            ? '1px solid rgba(255, 82, 82, 0.8)' 
            : '1px solid rgba(0, 230, 118, 0.8)'
        }}>
          <Typography 
            variant="subtitle2" 
            color={data.type === 'query' ? '#ff5252' : '#00e676'}
            sx={{ fontWeight: 'bold' }}
          >
            {data.type === 'query' ? 'YOUR QUERY' : 'Document ID: ' + data.id}
          </Typography>
          <Typography variant="body2" sx={{ mt: 1, color: '#fff' }}>
            {data.text}
          </Typography>
        </Paper>
      );
    }
    return null;
  };
  
  const SimilarityTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <Paper sx={{ 
          p: 2, 
          maxWidth: 300, 
          bgcolor: 'rgba(0, 0, 0, 0.85)',
          border: '1px solid rgba(0, 230, 118, 0.8)'
        }}>
          <Typography 
            variant="subtitle2" 
            color="#00e676"
            sx={{ fontWeight: 'bold' }}
          >
            Document ID: {data.id}
          </Typography>
          <Typography variant="body2" sx={{ mt: 1, color: '#fff' }}>
            {data.text}
          </Typography>
          <Typography variant="body2" sx={{ mt: 1, color: '#00e676', fontWeight: 'bold' }}>
            Similarity Score: {data.similarity.toFixed(4)}
          </Typography>
        </Paper>
      );
    }
    return null;
  };

  const getBarColor = (score) => {
    if (score > 0.8) return '#00E676'; // High similarity
    if (score > 0.6) return '#76FF03'; // Good similarity
    if (score > 0.4) return '#FFEA00'; // Moderate similarity
    if (score > 0.2) return '#FF9100'; // Low similarity
    return '#FF1744'; // Very low similarity
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
          Vector Space Explorer
        </Typography>
        
        <Typography variant="body1" sx={{ mb: 3 }}>
          Enter a query to visualize its position in the semantic vector space relative to the uploaded documents.
        </Typography>
        
        <Box sx={{ mb: 3, display: 'flex', gap: 2 }}>
          <TextField
            fullWidth
            label="Enter your query"
            value={query}
            onChange={handleQueryChange}
            variant="outlined"
            sx={{
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
          />
          <Button 
            variant="contained" 
            onClick={handleVisualize}
            disabled={loading || !query.trim()}
            sx={{ 
              minWidth: '120px',
              borderRadius: '8px',
            }}
            className="glow-effect"
          >
            {loading ? 'Processing...' : 'Visualize'}
          </Button>
        </Box>
        
        {loading && (
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
        
        {embeddingData && (
          <>
            <Paper sx={{ 
              p: 2, 
              height: 450, 
              border: '1px solid rgba(0, 230, 118, 0.2)',
              bgcolor: 'rgba(0, 0, 0, 0.2)'
            }}>
              <Typography 
                variant="h6" 
                gutterBottom
                sx={{ color: theme.palette.primary.light, mb: 2 }}
              >
                2D Embedding Visualization
              </Typography>
              
              <ResponsiveContainer width="100%" height="90%">
                <ScatterChart
                  margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    type="number" 
                    dataKey="x" 
                    name="Dimension 1" 
                    tick={{ fill: '#ccc' }}
                    label={{ 
                      value: 'Dimension 1', 
                      position: 'insideBottom', 
                      fill: '#ccc', 
                      offset: -5 
                    }}
                  />
                  <YAxis 
                    type="number" 
                    dataKey="y" 
                    name="Dimension 2"
                    tick={{ fill: '#ccc' }}
                    label={{ 
                      value: 'Dimension 2', 
                      angle: -90, 
                      position: 'insideLeft', 
                      fill: '#ccc',
                      offset: -5
                    }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Scatter 
                    name="Documents" 
                    data={embeddingData.filter(item => item.type === 'document')} 
                    fill="#00E676"
                    opacity={0.7}
                  />
                  <Scatter 
                    name="Your Query" 
                    data={embeddingData.filter(item => item.type === 'query')} 
                    fill="#FF5252"
                    shape="star"
                    opacity={1}
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </Paper>
            
            <Box sx={{ mt: 2, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                Points that are closer together are more semantically similar.
                <br />
                The visualization uses t-SNE dimensionality reduction to project high-dimensional embeddings to 2D space.
              </Typography>
            </Box>
            
            {similarityData && similarityData.length > 0 && (
              <>
                <Divider sx={{ my: 4, borderColor: 'rgba(0, 230, 118, 0.2)' }} />
                
                <Paper sx={{ 
                  p: 2, 
                  height: 400, 
                  mt: 3,
                  border: '1px solid rgba(0, 230, 118, 0.2)',
                  bgcolor: 'rgba(0, 0, 0, 0.2)'
                }}>
                  <Typography 
                    variant="h6" 
                    gutterBottom
                    sx={{ color: theme.palette.primary.light, mb: 2 }}
                  >
                    Cosine Similarity Scores
                  </Typography>
                  
                  <ResponsiveContainer width="100%" height="85%">
                    <BarChart
                      data={similarityData.slice(0, 10)} // Show only top 10
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis 
                        type="number" 
                        domain={[0, 1]} 
                        tick={{ fill: '#ccc' }}
                      />
                      <YAxis 
                        dataKey="id" 
                        type="category" 
                        tick={{ fill: '#ccc' }}
                        width={100}
                      />
                      <Tooltip content={<SimilarityTooltip />} />
                      <Legend />
                      <Bar 
                        dataKey="similarity" 
                        name="Similarity Score"
                        minPointSize={2}
                        isAnimationActive={true}
                      >
                        {similarityData.slice(0, 10).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={getBarColor(entry.similarity)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Paper>
                
                <Box sx={{ mt: 2, textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary">
                    Higher cosine similarity scores indicate greater semantic similarity between your query and the document.
                    <br />
                    Only the top 10 most similar documents are shown.
                  </Typography>
                </Box>
              </>
            )}
          </>
        )}
      </Paper>
    </motion.div>
  );
};

export default EmbeddingVisualizer; 