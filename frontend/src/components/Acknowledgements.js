import React from 'react';
import { Box, Typography, Paper, Grid, Link, Chip, Divider } from '@mui/material';
import { motion } from 'framer-motion';

const Acknowledgements = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        duration: 0.5,
        when: 'beforeChildren',
        staggerChildren: 0.1
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

  const technologies = [
    {
      category: 'Frontend Framework',
      items: [
        { name: 'React', url: 'https://reactjs.org/' },
        { name: 'Material UI', url: 'https://mui.com/' },
        { name: 'Framer Motion', url: 'https://www.framer.com/motion/' },
        { name: 'React Router', url: 'https://reactrouter.com/' }
      ]
    },
    {
      category: 'Backend Technologies',
      items: [
        { name: 'Flask', url: 'https://flask.palletsprojects.com/' },
        { name: 'Flask-CORS', url: 'https://flask-cors.readthedocs.io/' },
        { name: 'LlamaIndex', url: 'https://www.llamaindex.ai/' },
        { name: 'Pandas', url: 'https://pandas.pydata.org/' }
      ]
    },
    {
      category: 'UI Components',
      items: [
        { name: 'Material Icons', url: 'https://mui.com/material-ui/material-icons/' },
        { name: 'Material Data Grid', url: 'https://mui.com/x/react-data-grid/' },
        { name: 'Custom Cyberpunk Theme', url: '#' }
      ]
    },
    {
      category: 'Data Visualization',
      items: [
        { name: 'Material UI Tables', url: 'https://mui.com/material-ui/react-table/' },
        { name: 'CSS Animations', url: '#' }
      ]
    }
  ];

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <Paper 
        sx={{ 
          p: 4, 
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
          sx={{ mb: 4 }}
        >
          Acknowledgements
        </Typography>
        
        <Typography variant="body1" paragraph sx={{ mb: 4 }}>
          This application was built using modern web technologies and libraries. 
          We'd like to acknowledge the following tools and components that made this project possible.
        </Typography>
        
        <Grid container spacing={4}>
          {technologies.map((tech, index) => (
            <Grid item xs={12} md={6} key={index}>
              <motion.div variants={itemVariants}>
                <Paper 
                  sx={{ 
                    p: 3, 
                    height: '100%',
                    bgcolor: 'rgba(0, 0, 0, 0.2)',
                    border: '1px solid rgba(0, 230, 118, 0.2)',
                  }}
                  className="animated-border"
                >
                  <Typography 
                    variant="h6" 
                    component="h2" 
                    gutterBottom 
                    color="primary"
                    sx={{ 
                      fontWeight: 'bold',
                      borderBottom: '2px solid rgba(0, 230, 118, 0.3)',
                      pb: 1,
                      mb: 2
                    }}
                  >
                    {tech.category}
                  </Typography>
                  
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                    {tech.items.map((item, idx) => (
                      <Chip
                        key={idx}
                        label={item.name}
                        component={Link}
                        href={item.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        sx={{ 
                          bgcolor: 'rgba(0, 230, 118, 0.1)',
                          color: '#00E676',
                          border: '1px solid rgba(0, 230, 118, 0.3)',
                          '&:hover': {
                            bgcolor: 'rgba(0, 230, 118, 0.2)',
                            boxShadow: '0 0 8px rgba(0, 230, 118, 0.5)',
                          }
                        }}
                        className="glow-effect"
                        clickable
                      />
                    ))}
                  </Box>
                </Paper>
              </motion.div>
            </Grid>
          ))}
        </Grid>
        
        <Box sx={{ mt: 5, pt: 2, borderTop: '1px solid rgba(0, 230, 118, 0.2)' }}>
          <Typography variant="body2" color="text.secondary" align="center">
            Created with ❤️ and ☕. The code is available under an open-source license.
          </Typography>
          <Typography variant="body2" color="primary" align="center" sx={{ mt: 1 }}>
            © {new Date().getFullYear()} LlamaIndex CSV Analyzer
          </Typography>
        </Box>
      </Paper>
    </motion.div>
  );
};

export default Acknowledgements; 