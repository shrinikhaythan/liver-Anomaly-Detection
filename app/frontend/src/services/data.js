// Dummy medical data for liver CT anomaly detection system
// Replace with actual API calls in production

export const patientData = {
  patientName: "John Smith",
  patientId: "CT-2024-0891",
  studyDate: "2024-12-15",
  scanType: "Abdomen CT with Contrast"
};

export const processingSteps = [
  { id: 1, name: "Preprocessing", status: "completed", duration: "2.3s" },
  { id: 2, name: "CNN Liver Segmentation", status: "completed", duration: "5.7s" },
  { id: 3, name: "Cropping + Diffusion Reconstruction", status: "completed", duration: "12.4s" },
  { id: 4, name: "Heatmap Generation", status: "completed", duration: "3.1s" },
  { id: 5, name: "Traffic-Light Alert + Report", status: "completed", duration: "1.2s" }
];

export const ctSlices = [
  {
    sliceId: "S001",
    anomalyScore: 95,
    flagStatus: "red",
    findings: "Large hypodense lesion in segment VII, suspicious for metastatic disease",
    originalImage: "https://images.unsplash.com/photo-1715529134972-221f23a6c701?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwQ1QlMjBzY2FuJTIwbGl2ZXJ8ZW58MXx8fHwxNzU3NDExOTY0fDA&ixlib=rb-4.1.0&q=80&w=300&utm_source=figma&utm_medium=referral",
    reconstructedImage: "https://images.unsplash.com/photo-1715529134972-221f23a6c701?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwQ1QlMjBzY2FuJTIwbGl2ZXJ8ZW58MXx8fHwxNzU3NDExOTY0fDA&ixlib=rb-4.1.0&q=80&w=300&utm_source=figma&utm_medium=referral",
    heatmapImage: "https://images.unsplash.com/photo-1715529134972-221f23a6c701?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwQ1QlMjBzY2FuJTIwbGl2ZXJ8ZW58MXx8fHwxNzU3NDExOTY0fDA&ixlib=rb-4.1.0&q=80&w=300&utm_source=figma&utm_medium=referral"
  },
  {
    sliceId: "S002", 
    anomalyScore: 72,
    flagStatus: "yellow",
    findings: "Small hypodense area in segment IV, requires follow-up imaging",
    originalImage: "https://images.unsplash.com/photo-1715529134972-221f23a6c701?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwQ1QlMjBzY2FuJTIwbGl2ZXJ8ZW58MXx8fHwxNzU3NDExOTY0fDA&ixlib=rb-4.1.0&q=80&w=300&utm_source=figma&utm_medium=referral",
    reconstructedImage: "https://images.unsplash.com/photo-1715529134972-221f23a6c701?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwQ1QlMjBzY2FuJTIwbGl2ZXJ8ZW58MXx8fHwxNzU3NDExOTY0fDA&ixlib=rb-4.1.0&q=80&w=300&utm_source=figma&utm_medium=referral", 
    heatmapImage: "https://images.unsplash.com/photo-1715529134972-221f23a6c701?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwQ1QlMjBzY2FuJTIwbGl2ZXJ8ZW58MXx8fHwxNzU3NDExOTY0fDA&ixlib=rb-4.1.0&q=80&w=300&utm_source=figma&utm_medium=referral"
  },
  {
    sliceId: "S003",
    anomalyScore: 15,
    flagStatus: "green", 
    findings: "Normal liver parenchyma, no significant abnormalities detected",
    originalImage: "https://images.unsplash.com/photo-1715529134972-221f23a6c701?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwQ1QlMjBzY2FuJTIwbGl2ZXJ8ZW58MXx8fHwxNzU3NDExOTY0fDA&ixlib=rb-4.1.0&q=80&w=300&utm_source=figma&utm_medium=referral",
    reconstructedImage: "https://images.unsplash.com/photo-1715529134972-221f23a6c701?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwQ1QlMjBzY2FuJTIwbGl2ZXJ8ZW58MXx8fHwxNzU3NDExOTY0fDA&ixlib=rb-4.1.0&q=80&w=300&utm_source=figma&utm_medium=referral",
    heatmapImage: "https://images.unsplash.com/photo-1715529134972-221f23a6c701?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwQ1QlMjBzY2FuJTIwbGl2ZXJ8ZW58MXx8fHwxNzU3NDExOTY0fDA&ixlib=rb-4.1.0&q=80&w=300&utm_source=figma&utm_medium=referral"
  },
  {
    sliceId: "S004",
    anomalyScore: 88,
    flagStatus: "red",
    findings: "Multiple hypodense lesions throughout liver, concerning for metastatic spread",
    originalImage: "https://images.unsplash.com/photo-1715529134972-221f23a6c701?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwQ1QlMjBzY2FuJTIwbGl2ZXJ8ZW58MXx8fHwxNzU3NDExOTY0fDA&ixlib=rb-4.1.0&q=80&w=300&utm_source=figma&utm_medium=referral", 
    reconstructedImage: "https://images.unsplash.com/photo-1715529134972-221f23a6c701?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwQ1QlMjBzY2FuJTIwbGl2ZXJ8ZW58MXx8fHwxNzU3NDExOTY0fDA&ixlib=rb-4.1.0&q=80&w=300&utm_source=figma&utm_medium=referral",
    heatmapImage: "https://images.unsplash.com/photo-1715529134972-221f23a6c701?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwQ1QlMjBzY2FuJTIwbGl2ZXJ8ZW58MXx8fHwxNzU3NDExOTY0fDA&ixlib=rb-4.1.0&q=80&w=300&utm_source=figma&utm_medium=referral"
  },
  {
    sliceId: "S005",
    anomalyScore: 45,
    flagStatus: "yellow",
    findings: "Mild hepatic steatosis, borderline findings",
    originalImage: "https://images.unsplash.com/photo-1715529134972-221f23a6c701?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwQ1QlMjBzY2FuJTIwbGl2ZXJ8ZW58MXx8fHwxNzU3NDExOTY0fDA&ixlib=rb-4.1.0&q=80&w=300&utm_source=figma&utm_medium=referral",
    reconstructedImage: "https://images.unsplash.com/photo-1715529134972-221f23a6c701?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwQ1QlMjBzY2FuJTIwbGl2ZXJ8ZW58MXx8fHwxNzU3NDExOTY0fDA&ixlib=rb-4.1.0&q=80&w=300&utm_source=figma&utm_medium=referral",
    heatmapImage: "https://images.unsplash.com/photo-1715529134972-221f23a6c701?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwQ1QlMjBzY2FuJTIwbGl2ZXJ8ZW58MXx8fHwxNzU3NDExOTY0fDA&ixlib=rb-4.1.0&q=80&w=300&utm_source=figma&utm_medium=referral"
  }
];

export const uploadStats = {
  totalSlices: 156,
  processedSlices: 156,
  anomalousSlices: 23,
  processingTime: "24.7s"
};

export const overallAssessment = {
  maxAnomalyScore: 95,
  flagStatus: "red",
  summary: "Multiple concerning lesions detected requiring immediate clinical correlation"
};