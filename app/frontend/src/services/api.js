// API service for communicating with the Flask backend
const API_BASE_URL = 'http://localhost:5000';

class ApiService {
  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  // Upload CT scan file
  async uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${this.baseUrl}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Upload error:', error);
      throw error;
    }
  }

  // Process uploaded scan
  async processScan(scanId) {
    try {
      const response = await fetch(`${this.baseUrl}/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ scan_id: scanId }),
      });

      if (!response.ok) {
        throw new Error(`Processing failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Processing error:', error);
      throw error;
    }
  }

  // Get processing progress
  async getProgress(scanId) {
    try {
      const response = await fetch(`${this.baseUrl}/progress/${scanId}`);
      
      if (!response.ok) {
        throw new Error(`Progress check failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Progress check error:', error);
      throw error;
    }
  }

  // Get report for processed scan
  async getReport(scanId) {
    try {
      const response = await fetch(`${this.baseUrl}/report/${scanId}`);
      
      if (!response.ok) {
        throw new Error(`Report fetch failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Report fetch error:', error);
      throw error;
    }
  }

  // Get result image URL
  getResultImageUrl(imagePath) {
    return `${this.baseUrl}${imagePath}`;
  }

  // Health check
  async healthCheck() {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      return response.ok;
    } catch (error) {
      return false;
    }
  }
}

export const apiService = new ApiService();
export default apiService;
