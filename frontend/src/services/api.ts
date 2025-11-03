import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001';

export const fetchHackAnalysisData = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/hack_analysis/main`);
    return response.data;
  } catch (error) {
    console.error('Error fetching hack analysis data:', error);
    throw error;
  }
};

export const fetchSarsList = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/hack_analysis/sars-list`, { responseType: 'stream' });
    return JSON.parse(response.data);
  } catch (error) {
    console.error('Error fetching sars data:', error);
    throw error;
  }
};

export const fetchNetworkVisualizationDetails = async (hackName: string = "") => {
  try {
    const response = await axios.get(`${API_BASE_URL}/hack_analysis/network-visualization/${hackName}`, { responseType: 'stream' });
    return JSON.parse(response.data);
  } catch (error) {
    console.error('Error fetching sars data:', error);
    throw error;
  }
};

export const fetchSarsFilteredList = async (hackName: string = "") => {
  try {
    const response = await axios.get(`${API_BASE_URL}/hack_analysis/sars-list/${hackName}`, { responseType: 'stream' });
    return JSON.parse(response.data);
  } catch (error) {
    console.error('Error fetching sars data:', error);
    throw error;
  }
};

export const fetchNetworkExplanationSummary = async (hackName: string = "") => {
  try {
    const response = await axios.get(`${API_BASE_URL}/hack_analysis/offshore-analysis-summary/${hackName}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching explanation summary data:', error);
    throw error;
  }
};

export const fetchNetworkDiagram = async (hackName: string = "") => {
  try {
    const response = await axios.get(`${API_BASE_URL}/hack_analysis/offshore-analysis-visualization/${hackName}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching explanation summary data:', error);
    throw error;
  }
};

export const fetchReportFile = async (name: string): Promise<void> => {
  try {
    const response = await axios.get<Blob>(`${API_BASE_URL}/report/${name}`, {
      responseType: "blob", // ensure binary data
    });

    // Create a Blob URL from the response
    const blob = new Blob([response.data], {
      type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    });

    const url = window.URL.createObjectURL(blob);

    // Create a temporary anchor element
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `${name}.docx`); // file name with extension
    document.body.appendChild(link);
    link.click();
    link.remove();

    // Release memory
    window.URL.revokeObjectURL(url);
  } catch (error) {
    console.error("Failed to download report:", error);
  }
}