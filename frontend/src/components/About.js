import React from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Tooltip, 
  Divider, 
  Link, 
  Avatar,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow 
} from '@mui/material';
import { motion } from 'framer-motion';
import CodeIcon from '@mui/icons-material/Code';
import StorageIcon from '@mui/icons-material/Storage';
import MemoryIcon from '@mui/icons-material/Memory';
import AssessmentIcon from '@mui/icons-material/Assessment';
import GitHubIcon from '@mui/icons-material/GitHub';
import StarIcon from '@mui/icons-material/Star';
import BugReportIcon from '@mui/icons-material/BugReport';
import SchoolIcon from '@mui/icons-material/School';

const About = () => {
  // Animation variants
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

  // Team data with COD-style stats
  const team = [
    {
      name: 'Warren Low',
      role: 'Y2 CS',
      tagline: 'AI SPECIALIST',
      contributions: ['AI Engineer', 'RAG Developer', 'Frontend Engineer'],
      color: '#00E676',
      icon: <MemoryIcon sx={{ fontSize: 30, color: '#00E676', opacity: 0.8 }} />,
      github: 'DESU-CLUB',
      avatar: 'WL',
      image: 'https://media.tenor.com/X3CiuAO5T_YAAAAM/happy-excited.gif',
      useImage: true,
      stats: [
        { name: 'PR RATIO', value: '8.5', icon: <StarIcon sx={{ fontSize: 16 }} /> },
        { name: 'CODE SCORE', value: '1337', icon: <CodeIcon sx={{ fontSize: 16 }} /> },
        { name: 'BUG COUNT', value: '9', icon: <BugReportIcon sx={{ fontSize: 16 }} /> }
      ],
      rank: 'PRESTIGE 3'
    },
    {
      name: 'Somneel Saha',
      role: 'Y2 CS',
      tagline: 'FRONTEND ACE',
      contributions: ['Frontend Engineer', 'Evals Engineer'],
      color: '#00BFFF',
      icon: <CodeIcon sx={{ fontSize: 30, color: '#00BFFF', opacity: 0.8 }} />,
      github: 'SomneelSaha2004',
      avatar: 'SS',
      image: 'https://media.tenor.com/MtVsnj-yQAMAAAAM/tralalero-tralala.gif',
      useImage: true,
      stats: [
        { name: 'PR RATIO', value: '7.2', icon: <StarIcon sx={{ fontSize: 16 }} /> },
        { name: 'CODE SCORE', value: '920', icon: <CodeIcon sx={{ fontSize: 16 }} /> },
        { name: 'BUG COUNT', value: '2', icon: <BugReportIcon sx={{ fontSize: 16 }} /> }
      ],
      rank: 'PRESTIGE 3'
    },
    {
      name: 'Yao Hejun',
      role: 'Y2 CS',
      tagline: 'EVALS EXPERT',
      contributions: ['Evals Engineer', 'Dataset Curator'],
      color: '#FFD700',
      icon: <AssessmentIcon sx={{ fontSize: 30, color: '#FFD700', opacity: 0.8 }} />,
      github: 'testing1234567891011121314',
      avatar: 'YH',
      image: '/photo_6307338987085349641_x.jpg',
      useImage: true,
      stats: [
        { name: 'PR RATIO', value: '6.9', icon: <StarIcon sx={{ fontSize: 16 }} /> },
        { name: 'CODE SCORE', value: '880', icon: <CodeIcon sx={{ fontSize: 16 }} /> },
        { name: 'BUG COUNT', value: '4', icon: <BugReportIcon sx={{ fontSize: 16 }} /> }
      ],
      rank: 'PRESTIGE 2'
    },
    {
      name: 'Mitra Reet',
      role: 'Y3 CS',
      tagline: 'DATA COMMANDER',
      contributions: ['Dataset Curator'],
      color: '#FF4500',
      icon: <StorageIcon sx={{ fontSize: 30, color: '#FF4500', opacity: 0.8 }} />,
      github: 'reetmitra',
      avatar: 'MR',
      useImage: false,
      stats: [
        { name: 'PR RATIO', value: '9.1', icon: <StarIcon sx={{ fontSize: 16 }} /> },
        { name: 'CODE SCORE', value: '1250', icon: <CodeIcon sx={{ fontSize: 16 }} /> },
        { name: 'BUG COUNT', value: '0', icon: <BugReportIcon sx={{ fontSize: 16 }} /> }
      ],
      rank: 'PRESTIGE 1'
    }
  ];

  // Custom tooltip content
  const ProfileTooltip = ({ member }) => (
    <Box sx={{ 
      width: 300, 
      p: 2, 
      bgcolor: 'rgba(0, 0, 0, 0.9)',
      border: `1px solid ${member.color}`,
      boxShadow: `0 0 15px ${member.color}60`,
      borderRadius: '4px',
    }}>
      {/* Image & Basic Info */}
      <Box sx={{ mb: 2 }}>
        {member.useImage ? (
          <Box 
            component="img" 
            src={member.image} 
            alt={member.name}
            sx={{
              width: '100%',
              height: 140,
              objectFit: 'cover',
              borderRadius: '4px',
              border: `1px solid ${member.color}80`,
              mb: 1
            }}
          />
        ) : (
          <Box
            sx={{
              width: '100%',
              height: 140,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '72px',
              color: member.color,
              background: `radial-gradient(circle, ${member.color}30 0%, rgba(0,0,0,0.7) 70%)`,
              borderRadius: '4px',
              border: `1px solid ${member.color}80`,
              mb: 1
            }}
          >
            {member.avatar}
          </Box>
        )}
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6" sx={{ color: member.color, fontWeight: 'bold' }}>
            {member.name}
          </Typography>
          <Box 
            sx={{ 
              bgcolor: `${member.color}20`, 
              px: 1, 
              py: 0.5, 
              borderRadius: '2px',
              border: `1px solid ${member.color}40` 
            }}
          >
            <Typography variant="caption" sx={{ color: 'white', fontWeight: 'bold' }}>
              {member.rank}
            </Typography>
          </Box>
        </Box>
        
        <Typography variant="body2" sx={{ color: 'white', mb: 1 }}>
          {member.tagline}
        </Typography>
      </Box>
      
      {/* Stats */}
      <Typography variant="caption" sx={{ color: member.color, fontWeight: 'bold' }}>
        STATS
      </Typography>
      <Divider sx={{ borderColor: `${member.color}40`, my: 0.5 }} />
      
      <Box sx={{ mb: 1.5 }}>
        {member.stats.map((stat, idx) => (
          <Box 
            key={idx} 
            sx={{ 
              display: 'flex', 
              justifyContent: 'space-between',
              alignItems: 'center',
              mb: 0.5,
              p: 0.75,
              bgcolor: `${member.color}10`,
              border: `1px solid ${member.color}30`,
              borderRadius: '2px'
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              {stat.icon}
              <Typography variant="caption" sx={{ ml: 0.5, color: 'rgba(255,255,255,0.8)' }}>
                {stat.name}
              </Typography>
            </Box>
            <Typography variant="caption" sx={{ fontWeight: 'bold', color: member.color }}>
              {stat.value}
            </Typography>
          </Box>
        ))}
      </Box>
      
      {/* Contributions */}
      <Typography variant="caption" sx={{ color: member.color, fontWeight: 'bold' }}>
        LOADOUT
      </Typography>
      <Divider sx={{ borderColor: `${member.color}40`, my: 0.5 }} />
      
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
        {member.contributions.map((contrib, idx) => (
          <Box 
            key={idx} 
            sx={{
              p: 0.5,
              bgcolor: `${member.color}20`,
              border: `1px solid ${member.color}40`,
              borderRadius: '2px',
              fontSize: '0.7rem',
              color: 'white',
              display: 'flex',
              alignItems: 'center'
            }}
          >
            {member.icon}
            <Typography variant="caption" sx={{ ml: 0.5 }}>
              {contrib}
            </Typography>
          </Box>
        ))}
      </Box>
    </Box>
  );

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      style={{ width: '100%', maxWidth: '1400px', margin: '0 auto' }}
    >
      <Paper 
        sx={{ 
          p: 4, 
          mb: 3,
          border: '1px solid rgba(0, 230, 118, 0.2)',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4)',
          width: '100%',
          bgcolor: 'rgba(0, 0, 0, 0.6)'
        }}
      >
        <Typography 
          variant="h4" 
          component="h1" 
          gutterBottom
          className="cyber-header"
          sx={{ 
            mb: 2,
            background: 'linear-gradient(90deg, #00E676, #69F0AE)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            textShadow: '0 0 10px rgba(0, 230, 118, 0.5)'
          }}
        >
          TEAM ROSTER
        </Typography>
        
        <Box 
          sx={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            mb: 3,
            p: 1,
            borderRadius: '4px',
            background: 'rgba(0, 230, 118, 0.1)',
          }}
        >
          <Typography variant="body2" sx={{ color: '#00E676' }}>
            PROJECT STATUS: COOKED
          </Typography>
          <Typography variant="body2" sx={{ color: '#00E676' }}>
            TEAM: CS3264 GROUP 23
          </Typography>
          <Typography variant="body2" sx={{ color: '#00E676' }}>
            PROJECT: ML DATA GENERATION
          </Typography>
        </Box>
        
        <TableContainer 
          component={Paper} 
          sx={{ 
            bgcolor: 'rgba(0, 0, 0, 0.4)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            backgroundImage: 'linear-gradient(rgba(0, 230, 118, 0.05), transparent)',
            boxShadow: 'none'
          }}
        >
          <Table sx={{ '& .MuiTableCell-root': { borderColor: 'rgba(255, 255, 255, 0.1)' } }}>
            <TableHead>
              <TableRow sx={{ 
                bgcolor: 'rgba(0, 0, 0, 0.6)',
                '& th': { color: 'rgba(255, 255, 255, 0.7)', fontWeight: 'bold' } 
              }}>
                <TableCell width={50}>#</TableCell>
                <TableCell>NAME</TableCell>
                <TableCell>RANK</TableCell>
                <TableCell>YEAR</TableCell>
                <TableCell align="right">GITHUB</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {team.map((member, index) => (
                <Tooltip
                  key={index}
                  title={<ProfileTooltip member={member} />}
                  arrow
                  placement="right"
                  componentsProps={{
                    tooltip: {
                      sx: { 
                        bgcolor: 'transparent',
                        p: 0,
                        '& .MuiTooltip-arrow': {
                          color: member.color
                        }
                      }
                    }
                  }}
                >
                  <motion.tr
                    variants={itemVariants}
                    component={TableRow}
                    sx={{
                      cursor: 'pointer',
                      position: 'relative',
                      '&:hover': {
                        bgcolor: `${member.color}20`,
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          left: 0,
                          top: 0,
                          width: '4px',
                          height: '100%',
                          bgcolor: member.color
                        }
                      }
                    }}
                  >
                    <TableCell>
                      <Avatar 
                        sx={{ 
                          bgcolor: `${member.color}30`, 
                          width: 32, 
                          height: 32,
                          border: `1px solid ${member.color}70`
                        }}
                      >
                        {index + 1}
                      </Avatar>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        {member.icon}
                        <Typography 
                          sx={{ 
                            color: member.color, 
                            fontWeight: 'bold',
                            ml: 1,
                            textShadow: `0 0 5px ${member.color}50`
                          }}
                        >
                          {member.name}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell sx={{ color: 'white' }}>
                      {member.rank}
                    </TableCell>
                    <TableCell sx={{ color: 'white' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <SchoolIcon sx={{ fontSize: 16, mr: 0.5, color: member.color }} />
                        {member.role}
                      </Box>
                    </TableCell>
                    <TableCell align="right">
                      <Link 
                        href={`https://github.com/${member.github}`}
                        target="_blank"
                        rel="noopener"
                        sx={{ 
                          color: 'rgba(255, 255, 255, 0.7)',
                          textDecoration: 'none',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'flex-end',
                          '&:hover': { color: member.color }
                        }}
                      >
                        @{member.github}
                        <GitHubIcon sx={{ ml: 1, fontSize: 18 }} />
                      </Link>
                    </TableCell>
                  </motion.tr>
                </Tooltip>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        
        <Typography 
          variant="caption" 
          sx={{ 
            display: 'block',
            textAlign: 'center',
            mt: 2,
            color: 'rgba(255, 255, 255, 0.5)' 
          }}
        >
          Hover over a player to view detailed stats and profile
        </Typography>
      </Paper>
    </motion.div>
  );
};

export default About; 