import React from 'react';
import { Paper, Typography, Box } from '@mui/material';
import { motion } from 'framer-motion';
import EmbeddingVisualizer from './EmbeddingVisualizer';

const Database = () => {
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
          Database Management
        </Typography>
        
        <EmbeddingVisualizer />
      </Paper>
    </motion.div>
  );
};

export default Database; 