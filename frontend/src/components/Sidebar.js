import React from 'react';
import { 
  Box, 
  Drawer, 
  List, 
  ListItem, 
  ListItemButton,
  ListItemIcon, 
  ListItemText, 
  IconButton,
  Typography,
  Divider,
  LinearProgress,
  Paper
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { motion } from 'framer-motion';
import { Link as RouterLink, useLocation } from 'react-router-dom';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import TableChartIcon from '@mui/icons-material/TableChart';
import DataObjectIcon from '@mui/icons-material/DataObject';
import StorageIcon from '@mui/icons-material/Storage';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import InfoIcon from '@mui/icons-material/Info';
import HomeIcon from '@mui/icons-material/Home';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';

const drawerWidth = 280;

const DrawerHeader = styled('div')(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  padding: theme.spacing(0, 1),
  ...theme.mixins.toolbar,
  justifyContent: 'flex-end',
}));

const Sidebar = ({ 
  isOpen, 
  toggleSidebar, 
  uploadResponse,
  isGenerating,
  generationProgress
}) => {
  const location = useLocation();
  
  // Menu items
  const menuItems = [
    { text: 'Home', icon: <HomeIcon />, path: '/' },
    { text: 'Data Explorer', icon: <TableChartIcon />, path: '/explorer' },
    { text: 'Analysis', icon: <AnalyticsIcon />, path: '/analysis' },
    { text: 'Data Generation', icon: <DataObjectIcon />, path: '/generation' },
    { text: 'Database', icon: <StorageIcon />, path: '/database' },
    { text: 'About Us', icon: <InfoIcon />, path: '/about' },
    { text: 'Acknowledgements', icon: <InfoIcon />, path: '/acknowledgements' }
  ];
  
  return (
    <Drawer
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          borderRight: '1px solid rgba(0, 230, 118, 0.2)',
          background: '#151515',
          boxShadow: isOpen ? '0 0 20px rgba(0, 230, 118, 0.15)' : 'none',
        },
      }}
      variant="persistent"
      anchor="left"
      open={isOpen}
      className="drawer-container"
    >
      <DrawerHeader>
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          width: '100%', 
          justifyContent: 'space-between',
          ml: 2 
        }}>
          <Typography 
            variant="h6" 
            component="div" 
            color="primary" 
            sx={{ 
              fontWeight: 'bold', 
              letterSpacing: '1px',
              textTransform: 'uppercase'
            }}
            className="cyber-header"
          >
            LlamaIndex
          </Typography>
          <IconButton onClick={toggleSidebar} className="neon-button">
            {isOpen ? <ChevronLeftIcon /> : <ChevronRightIcon />}
          </IconButton>
        </Box>
      </DrawerHeader>

      <Divider sx={{ borderColor: 'rgba(0, 230, 118, 0.2)' }} />

      <Box sx={{ p: 2 }}>
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          sx={{ 
            p: 2,
            backgroundColor: "rgba(0, 0, 0, 0.3)",
            borderRadius: "8px",
            border: "1px solid rgba(255, 255, 255, 0.1)"
          }}
        >
          {isGenerating ? (
            <>
              <Typography variant="subtitle2" gutterBottom sx={{ color: '#00E676' }}>
                Generating Data
              </Typography>
              <Box sx={{ width: '100%', mt: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" sx={{ color: 'rgba(0, 230, 118, 0.9)' }}>
                    Progress
                  </Typography>
                  <Typography variant="caption" sx={{ color: 'rgba(0, 230, 118, 0.9)' }}>
                    {Math.round(generationProgress || 0)}%
                  </Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={generationProgress || 0} 
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
              </Box>
            </>
          ) : (
            <>
              <Typography variant="subtitle2" gutterBottom sx={{ color: 'rgba(255, 255, 255, 0.9)' }}>
                Current Dataset
              </Typography>
              {uploadResponse ? (
                <Box 
                  component={Paper} 
                  sx={{ 
                    p: 1.5, 
                    mt: 1, 
                    display: 'flex', 
                    alignItems: 'center',
                    bgcolor: 'rgba(0, 0, 0, 0.3)',
                    border: '1px solid rgba(255, 255, 255, 0.2)'
                  }}
                >
                  <InsertDriveFileIcon sx={{ color: 'rgba(255, 255, 255, 0.8)', mr: 1, fontSize: '1.2rem' }} />
                  <Box>
                    <Typography variant="body2" sx={{ wordBreak: 'break-word' }}>
                      {uploadResponse.filename || 
                       (uploadResponse.current_file && typeof uploadResponse.current_file === 'string' 
                        ? uploadResponse.current_file.split('/').pop() 
                        : 'Dataset')}
                    </Typography>
                    <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                      {uploadResponse.rowCount || uploadResponse.rows || 0} rows · 
                      {uploadResponse.columnCount || (uploadResponse.columns ? uploadResponse.columns.length : 0)} columns
                    </Typography>
                  </Box>
                </Box>
              ) : (
                <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.9)', mt: 1 }}>
                  No CSV files added yet
                </Typography>
              )}
            </>
          )}
        </motion.div>
      </Box>

      <Divider sx={{ borderColor: 'rgba(0, 230, 118, 0.2)', my: 2 }} />

      <List>
        {menuItems.map((item, index) => {
          const isActive = location.pathname === item.path;
          
          return (
            <ListItem key={index} disablePadding>
              <ListItemButton 
                component={RouterLink}
                to={item.path}
                sx={{
                  backgroundColor: isActive ? 'rgba(0, 230, 118, 0.15)' : 'transparent',
                  borderLeft: isActive ? '4px solid #00E676' : '4px solid transparent',
                  pl: isActive ? 2.5 : 3,
                  '&:hover': {
                    backgroundColor: 'rgba(0, 230, 118, 0.08)',
                  },
                }}
              >
                <ListItemIcon sx={{ color: isActive ? '#00E676' : 'rgba(0, 230, 118, 0.7)' }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText 
                  primary={item.text} 
                  sx={{ 
                    color: '#fff',
                    '& .MuiTypography-root': {
                      fontWeight: isActive ? 700 : 500,
                    }
                  }} 
                />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>
    </Drawer>
  );
};

export default Sidebar; 