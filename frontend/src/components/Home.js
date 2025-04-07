import React from 'react';
import { 
  Typography, 
  Box, 
  Grid, 
  Card,
  CardContent,
  Button,
  Paper,
  Stack,
  Divider,
  useTheme,
  Container
} from '@mui/material';
import { motion } from 'framer-motion';
import DataIcon from '@mui/icons-material/Storage';
import AnalyticsIcon from '@mui/icons-material/BarChart';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import SpeedIcon from '@mui/icons-material/Speed';
import SecurityIcon from '@mui/icons-material/Security';
import { Link } from 'react-router-dom';

const Home = () => {
  const theme = useTheme();
  
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

  // Features list
  const features = [
    {
      icon: <AutoAwesomeIcon fontSize="large" sx={{ color: '#00E676' }} />,
      title: "AI-Powered Data Synthesis",
      description: "Generate realistic, synthetic datasets using state-of-the-art GPT-4o Mini. Perfect for testing, development, and training without compromising privacy."
    },
    {
      icon: <AnalyticsIcon fontSize="large" sx={{ color: '#00E676' }} />,
      title: "Interactive Analysis",
      description: "Visualize and explore your data through intuitive charts and graphs. Ask questions about your data and get instant insights."
    },
    {
      icon: <DataIcon fontSize="large" sx={{ color: '#00E676' }} />,
      title: "Flexible Data Explorer",
      description: "Upload your own CSV files or generate synthetic data. Our platform adapts to your needs and helps you make sense of complex datasets."
    },
    {
      icon: <SpeedIcon fontSize="large" sx={{ color: '#00E676' }} />,
      title: "Real-time Interaction",
      description: "Engage with your data while processing is ongoing. No need to wait - start exploring insights as they become available."
    },
    {
      icon: <SecurityIcon fontSize="large" sx={{ color: '#00E676' }} />,
      title: "Privacy-First Approach",
      description: "Generate synthetic data that preserves statistical properties without exposing sensitive information from original datasets."
    }
  ];
  
  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      style={{ 
        width: '100%',
        padding: 0,
        margin: 0,
        boxSizing: 'border-box',
        touchAction: 'pan-y',
        minHeight: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center'
      }}
    >
      {/* Hero Section */}
      <Paper 
        elevation={0}
        sx={{ 
          p: { xs: 3, md: 5 }, 
          mb: 4,
          borderRadius: '16px',
          backgroundImage: 'linear-gradient(135deg, rgba(0, 0, 0, 0.7) 0%, rgba(0, 40, 20, 0.8) 100%)',
          border: '1px solid rgba(0, 230, 118, 0.2)',
          boxShadow: '0 4px 30px rgba(0, 230, 118, 0.15)',
          width: '100%',
          mx: 'auto'
        }}
      >
        <motion.div variants={itemVariants}>
          <Grid container spacing={4} alignItems="center" justifyContent="center">
            <Grid item xs={12} md={7}>
              <Typography 
                variant="h2" 
                component="h1"
                className="cyber-header"
                sx={{ 
                  fontWeight: 700,
                  mb: 2,
                  background: 'linear-gradient(90deg, #00E676 0%, #00B0FF 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                Synthetic Data Platform
              </Typography>
              
              <Typography 
                variant="h5" 
                color="text.secondary" 
                sx={{ 
                  mb: 3,
                  fontWeight: 300,
                  lineHeight: 1.5
                }}
              >
                Generate, explore, and analyze synthetic data powered by GPT-4o Mini
              </Typography>
              
              <Typography 
                variant="body1" 
                color="text.secondary"
                sx={{ mb: 4, maxWidth: 580 }}
              >
                Our platform enables you to create high-quality synthetic datasets, explore data interactively,
                and get insights through natural language queries - all while maintaining privacy and security.
              </Typography>
              
              <Stack direction="row" spacing={2}>
                <Button 
                  variant="contained" 
                  component={Link}
                  to="/generation"
                  size="large"
                  sx={{ 
                    px: 4,
                    py: 1.5,
                    borderRadius: '8px',
                    background: 'linear-gradient(90deg, #00C853, #00E676)',
                    '&:hover': {
                      background: 'linear-gradient(90deg, #00B34A, #00D26A)',
                    },
                    boxShadow: '0 4px 14px rgba(0, 230, 118, 0.3)'
                  }}
                  className="glow-effect"
                >
                  Generate Data
                </Button>
                
                <Button 
                  variant="outlined"
                  component={Link}
                  to="/explorer"
                  size="large"
                  sx={{ 
                    px: 4,
                    py: 1.5,
                    borderRadius: '8px',
                    borderColor: 'rgba(0, 230, 118, 0.5)',
                    color: '#00E676',
                    '&:hover': {
                      borderColor: '#00E676',
                      backgroundColor: 'rgba(0, 230, 118, 0.04)'
                    }
                  }}
                >
                  Explore Data
                </Button>
              </Stack>
            </Grid>
            
            <Grid item xs={12} md={5}>
              <Box
                sx={{
                  width: '100%',
                  height: '300px',
                  display: { xs: 'none', md: 'flex' },
                  margin: '0 auto',
                  justifyContent: 'center',
                  alignItems: 'center',
                  color: '#00E676',
                  border: '1px solid rgba(0, 230, 118, 0.2)',
                  borderRadius: '16px',
                  background: 'rgba(0, 0, 0, 0.2)',
                  boxShadow: '0 6px 16px rgba(0, 230, 118, 0.15)'
                }}
              >
                <AnalyticsIcon sx={{ fontSize: 120, opacity: 0.8 }} />
              </Box>
            </Grid>
          </Grid>
        </motion.div>
      </Paper>
      
      {/* Features Section */}
      <Typography 
        variant="h4" 
        component="h2"
        className="cyber-header"
        sx={{ 
          mb: 4,
          textAlign: 'center',
          position: 'relative',
          width: '100%',
          '&:after': {
            content: '""',
            position: 'absolute',
            bottom: -10,
            left: '50%',
            transform: 'translateX(-50%)',
            width: 80,
            height: 3,
            background: 'linear-gradient(90deg, #00E676, transparent)',
            borderRadius: 2
          }
        }}
      >
        Key Features
      </Typography>
      
      <Grid container spacing={3} justifyContent="center" sx={{ 
        width: '100%',
        maxWidth: '1200px',
        m: 0,
        p: 0,
        mx: 'auto'
      }}>
        {features.map((feature, index) => (
          <Grid item key={index} sx={{ 
            width: { xs: '100%', sm: '300px' },
            padding: 2
          }}>
            <motion.div 
              variants={itemVariants}
              whileHover={{ y: -5, transition: { duration: 0.2 } }}
              style={{ height: '100%', width: '100%' }}
            >
              <Card 
                sx={{ 
                  height: '100%',
                  width: '100%',
                  bgcolor: 'rgba(0, 0, 0, 0.3)',
                  border: '1px solid rgba(0, 230, 118, 0.15)',
                  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    borderColor: 'rgba(0, 230, 118, 0.3)',
                    boxShadow: '0 8px 30px rgba(0, 230, 118, 0.15)',
                    bgcolor: 'rgba(0, 20, 10, 0.4)'
                  },
                  display: 'flex',
                  flexDirection: 'column'
                }}
              >
                <CardContent sx={{ p: 3, flex: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    {feature.icon}
                    <Typography 
                      variant="h6" 
                      component="h3" 
                      sx={{ ml: 1.5, color: '#00E676' }}
                    >
                      {feature.title}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {feature.description}
                  </Typography>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        ))}
      </Grid>
      
      {/* How It Works Section */}
      <Box sx={{ mt: 6, mb: 4, width: '100%', mx: 'auto' }}>
        <Typography 
          variant="h4" 
          component="h2"
          className="cyber-header"
          sx={{ 
            mb: 4,
            textAlign: 'center',
            position: 'relative',
            width: '100%',
            '&:after': {
              content: '""',
              position: 'absolute',
              bottom: -10,
              left: '50%',
              transform: 'translateX(-50%)',
              width: 80,
              height: 3,
              background: 'linear-gradient(90deg, #00E676, transparent)',
              borderRadius: 2
            }
          }}
        >
          How It Works
        </Typography>
        
        <Paper 
          sx={{ 
            p: 4,
            borderRadius: '12px',
            bgcolor: 'rgba(0, 0, 0, 0.2)',
            border: '1px solid rgba(0, 230, 118, 0.2)',
            width: '100%',
            maxWidth: '1200px',
            mx: 'auto'
          }}
        >
          <Grid container spacing={3} justifyContent="center">
            <Grid item sx={{ 
              width: { xs: '100%', sm: '300px' },
              padding: 2
            }}>
              <motion.div variants={itemVariants} style={{ height: '100%', width: '100%' }}>
                <Box sx={{ textAlign: 'center', p: 2, height: '100%', width: '100%' }}>
                  <Typography 
                    variant="h1" 
                    sx={{ 
                      fontWeight: 900, 
                      color: 'rgba(0, 230, 118, 0.2)',
                      fontSize: '5rem'
                    }}
                  >
                    1
                  </Typography>
                  <Typography variant="h6" sx={{ mb: 1, color: '#00E676' }}>
                    Generate or Upload
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Choose to generate synthetic data with GPT-4o Mini or upload your own CSV file for analysis
                  </Typography>
                </Box>
              </motion.div>
            </Grid>
            
            <Grid item sx={{ 
              width: { xs: '100%', sm: '300px' },
              padding: 2
            }}>
              <motion.div variants={itemVariants} style={{ height: '100%', width: '100%' }}>
                <Box sx={{ textAlign: 'center', p: 2, height: '100%', width: '100%' }}>
                  <Typography 
                    variant="h1" 
                    sx={{ 
                      fontWeight: 900, 
                      color: 'rgba(0, 230, 118, 0.2)',
                      fontSize: '5rem'
                    }}
                  >
                    2
                  </Typography>
                  <Typography variant="h6" sx={{ mb: 1, color: '#00E676' }}>
                    Explore & Visualize
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    View interactive visualizations, statistics, and insights from your data through our analysis tools
                  </Typography>
                </Box>
              </motion.div>
            </Grid>
            
            <Grid item sx={{ 
              width: { xs: '100%', sm: '300px' },
              padding: 2
            }}>
              <motion.div variants={itemVariants} style={{ height: '100%', width: '100%' }}>
                <Box sx={{ textAlign: 'center', p: 2, height: '100%', width: '100%' }}>
                  <Typography 
                    variant="h1" 
                    sx={{ 
                      fontWeight: 900, 
                      color: 'rgba(0, 230, 118, 0.2)',
                      fontSize: '5rem'
                    }}
                  >
                    3
                  </Typography>
                  <Typography variant="h6" sx={{ mb: 1, color: '#00E676' }}>
                    Query & Interact
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Ask questions in natural language about your data and receive detailed insights while processing
                  </Typography>
                </Box>
              </motion.div>
            </Grid>
          </Grid>
        </Paper>
      </Box>
      
      {/* Call to Action */}
      <Box sx={{ mt: 6, mb: 2, textAlign: 'center', width: '100%' }}>
        <motion.div variants={itemVariants}>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 300 }}>
            Ready to start working with data?
          </Typography>
          <Box sx={{ mt: 3 }}>
            <Button 
              variant="contained"
              component={Link}
              to="/generation"
              size="large"
              sx={{ 
                px: 5,
                py: 1.5,
                borderRadius: '8px',
                background: 'linear-gradient(90deg, #00C853, #00E676)',
                '&:hover': {
                  background: 'linear-gradient(90deg, #00B34A, #00D26A)',
                },
                boxShadow: '0 4px 14px rgba(0, 230, 118, 0.3)'
              }}
              className="glow-effect"
            >
              Get Started
            </Button>
          </Box>
        </motion.div>
      </Box>
    </motion.div>
  );
};

export default Home; 