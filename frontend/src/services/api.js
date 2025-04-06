import axios from 'axios';

const API_URL = 'http://localhost:5000';

// Flag to use dummy endpoints for testing - permanently set to false
const USE_DUMMY_ENDPOINTS = false;

// Configure axios defaults
axios.defaults.withCredentials = false;

const api = {
  // Upload a CSV file
  uploadFile: async (file) => {
    try {
      // Always use real endpoint - dummy endpoint code removed
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      return response.data;
    } catch (error) {
      console.error('Error in uploadFile:', error);
      if (error.response) {
        // Server responded with an error
        throw new Error(error.response.data.error || 'Failed to upload file');
      } else if (error.request) {
        // Network error
        throw new Error('Network error: Could not connect to server');
      } else {
        throw error;
      }
    }
  },
  
  // Delete the current CSV file
  deleteCurrentCSV: async () => {
    try {
      const response = await axios.post(`${API_URL}/delete_current_csv`);
      return response.data;
    } catch (error) {
      console.error('Error in deleteCurrentCSV:', error);
      if (error.response) {
        throw new Error(error.response.data.error || 'Failed to delete CSV file');
      } else if (error.request) {
        throw new Error('Network error: Could not connect to server');
      } else {
        throw error;
      }
    }
  },
  
  // Get generation status
  getGenerationStatus: async () => {
    try {
      // Always use real endpoint
      const response = await axios.get(`${API_URL}/generation_status`);
      return response.data;
    } catch (error) {
      console.error('Error in getGenerationStatus:', error);
      // Return default status in case of error to prevent UI breakage
      return {
        isGenerating: false,
        progress: 0,
        currentFile: null,
        error: error.message || 'Failed to fetch generation status',
        has_generated_data: false
      };
    }
  },
  
  // Stream analysis for a query
  streamAnalysis: async (query, onProgressCallback) => {
    try {
      // Always use real endpoint
      const response = await axios.post(`${API_URL}/stream_analysis`, {
        query
      }, {
        responseType: 'stream'
      });
      
      return response;
    } catch (error) {
      console.error('Error in streamAnalysis:', error);
      throw error;
    }
  },
  
  // Regular GET request
  get: async (endpoint) => {
    try {
      // Don't allow requests to dummy endpoints
      if (endpoint.includes('/dummy_')) {
        console.error(`Request to dummy endpoint ${endpoint} blocked`);
        throw new Error('Dummy endpoints are disabled');
      }
      
      console.log(`Making GET request to: ${API_URL}${endpoint}`);
      const response = await axios.get(`${API_URL}${endpoint}`);
      return response;
    } catch (error) {
      console.error(`Error in GET request to ${endpoint}:`, error);
      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        console.error('Server responded with error:', error.response.data);
        throw new Error(error.response.data.error || `Server error: ${error.response.status}`);
      } else if (error.request) {
        // The request was made but no response was received
        console.error('No response received from server');
        throw new Error('Network error: Could not connect to server');
      } else {
        // Something happened in setting up the request that triggered an Error
        throw error;
      }
    }
  },
  
  // Regular POST request
  post: async (endpoint, data) => {
    try {
      // Don't allow requests to dummy endpoints
      if (endpoint.includes('/dummy_')) {
        console.error(`Request to dummy endpoint ${endpoint} blocked`);
        throw new Error('Dummy endpoints are disabled');
      }
      
      console.log(`Making POST request to: ${API_URL}${endpoint}`, data);
      const response = await axios.post(`${API_URL}${endpoint}`, data);
      return response;
    } catch (error) {
      console.error(`Error in POST request to ${endpoint}:`, error);
      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        console.error('Server responded with error:', error.response.data);
        throw new Error(error.response.data.error || `Server error: ${error.response.status}`);
      } else if (error.request) {
        // The request was made but no response was received
        console.error('No response received from server');
        throw new Error('Network error: Could not connect to server');
      } else {
        // Something happened in setting up the request that triggered an Error
        throw error;
      }
    }
  },
  
  // Helper for handling streaming responses
  handleStreamResponse: async (response, onContentCallback, onInfoCallback, onErrorCallback, onCompleteCallback) => {
    const reader = response.data.getReader();
    const decoder = new TextDecoder();
    
    let accumulatedResponse = '';
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const textChunk = decoder.decode(value, { stream: true });
        const lines = textChunk.split('\n');
        
        for (const line of lines) {
          if (line.trim()) {
            try {
              const chunk = JSON.parse(line);
              
              switch (chunk.type) {
                case 'content':
                  accumulatedResponse += chunk.text;
                  if (onContentCallback) onContentCallback(accumulatedResponse, chunk.text);
                  break;
                case 'info':
                case 'progress':
                  if (onInfoCallback) onInfoCallback(chunk.message);
                  break;
                case 'complete':
                  if (onCompleteCallback) onCompleteCallback(chunk.message);
                  break;
                case 'error':
                  if (onErrorCallback) onErrorCallback(chunk.message);
                  break;
                default:
                  break;
              }
            } catch (e) {
              console.warn('Error parsing JSON from stream:', e, line);
            }
          }
        }
      }
      
      return accumulatedResponse;
    } catch (error) {
      console.error('Error handling stream response:', error);
      if (onErrorCallback) onErrorCallback('Error handling stream response');
      throw error;
    }
  },
  
  // Add a function to run pandas queries on the current CSV
  queryCSV: async (query) => {
    try {
      const response = await axios.post(`${API_URL}/query_csv`, { query });
      return response.data;
    } catch (error) {
      console.error('Error in queryCSV:', error);
      if (error.response) {
        throw new Error(error.response.data.error || 'Failed to execute query');
      } else if (error.request) {
        throw new Error('Network error: Could not connect to server');
      } else {
        throw error;
      }
    }
  },
  
  // Check CSV status
  checkCSVStatus: async () => {
    try {
      const response = await axios.get(`${API_URL}/check_csv_status`);
      return response.data;
    } catch (error) {
      console.error('Error in checkCSVStatus:', error);
      if (error.response) {
        throw new Error(error.response.data.error || 'Failed to check CSV status');
      } else if (error.request) {
        throw new Error('Network error: Could not connect to server');
      } else {
        throw error;
      }
    }
  },
  
  // Visualize embeddings
  visualizeEmbeddings: async (query) => {
    try {
      const response = await axios.post(`${API_URL}/visualize_embeddings`, { query });
      return response.data;
    } catch (error) {
      console.error('Error in visualizeEmbeddings:', error);
      if (error.response) {
        throw new Error(error.response.data.error || 'Failed to visualize embeddings');
      } else if (error.request) {
        throw new Error('Network error: Could not connect to server');
      } else {
        throw error;
      }
    }
  }
};

export default api; 