const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const fileList = document.getElementById("fileList");
const fileItems = document.getElementById("fileItems");
const uploadBtn = document.getElementById("uploadBtn");
const fileCount = document.getElementById("fileCount");
const totalSize = document.getElementById("totalSize");

let selectedFiles = [];

// ==================== TOAST NOTIFICATION SYSTEM ====================
function showToast(message, type = 'info', duration = 3000) {
  // Remove existing toasts
  const existingToasts = document.querySelectorAll('.custom-toast');
  existingToasts.forEach(toast => toast.remove());

  // Create toast element
  const toast = document.createElement('div');
  toast.className = `custom-toast custom-toast-${type}`;
  
  // Set icon based on type
  let icon = '';
  switch(type) {
    case 'success':
      icon = '✅';
      break;
    case 'error':
      icon = '❌';
      break;
    case 'warning':
      icon = '⚠️';
      break;
    case 'info':
      icon = 'ℹ️';
      break;
    default:
      icon = '📝';
  }
  
  toast.innerHTML = `
    <span class="toast-icon">${icon}</span>
    <span class="toast-message">${message}</span>
  `;
  
  document.body.appendChild(toast);
  
  // Trigger animation
  setTimeout(() => toast.classList.add('show'), 10);
  
  // Auto remove after duration
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ==================== FILE SELECTION HANDLERS ====================

// FIXED: Click to upload - Only trigger when clicking empty area
uploadArea.addEventListener("click", (e) => {
  // Don't trigger if clicking on interactive elements
  if (
    e.target.closest('.file-item') || 
    e.target.closest('.remove-btn') || 
    e.target.closest('.browse-btn') ||
    e.target.closest('.file-list') ||
    e.target.closest('.upload-text') ||
    e.target.tagName === 'BUTTON' ||
    e.target.tagName === 'INPUT'
  ) {
    return;
  }
  fileInput.click();
});

// Browse button click handler
const browseBtn = document.querySelector('.browse-btn');
if (browseBtn) {
  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });
}

// Drag and drop handlers
uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener("change", (e) => {
  handleFiles(e.target.files);
});

// ==================== FILE HANDLING ====================

function handleFiles(files) {
  const validExtensions = [
    "pdf", "doc", "docx", "txt", "xlsx", "csv",
    "ppt", "pptx", "jpg", "jpeg", "png", "gif"
  ];
  const maxSize = 10 * 1024 * 1024; // 10MB

  let hasInvalidFiles = false;
  let invalidFileNames = [];

  Array.from(files).forEach((file) => {
    const extension = file.name.split(".").pop().toLowerCase();
    
    if (!validExtensions.includes(extension)) {
      hasInvalidFiles = true;
      invalidFileNames.push(`${file.name} (invalid type)`);
      return;
    }

    if (file.size > maxSize) {
      hasInvalidFiles = true;
      invalidFileNames.push(`${file.name} (too large)`);
      return;
    }

    // Check for duplicates
    const isDuplicate = selectedFiles.some(f => f.name === file.name && f.size === file.size);
    if (!isDuplicate) {
      selectedFiles.push(file);
    }
  });

  if (hasInvalidFiles) {
    const errorMsg = invalidFileNames.length > 0 
      ? `Invalid files:\n${invalidFileNames.join('\n')}` 
      : 'Some files are invalid';
    showToast(errorMsg, 'error', 4000);
  }

  if (selectedFiles.length > 0) {
    displayFiles();
    updateStats();
  }
}

function displayFiles() {
  if (selectedFiles.length === 0) {
    fileList.classList.remove('show');
    uploadBtn.disabled = true;
    return;
  }

  fileList.classList.add('show');
  uploadBtn.disabled = false;
  fileItems.innerHTML = "";

  selectedFiles.forEach((file, index) => {
    const fileItem = document.createElement("div");
    fileItem.className = "file-item";

    const fileIcon = getFileIcon(file.name);
    const fileSize = formatFileSize(file.size);

    fileItem.innerHTML = `
      <div class="file-info">
        <div class="file-icon-small">${fileIcon}</div>
        <div class="file-details">
          <div class="file-name">${file.name}</div>
          <div class="file-size">${fileSize}</div>
        </div>
      </div>
      <div class="file-actions">
        <button class="remove-btn" onclick="removeFile(${index}); event.stopPropagation();">✕ Remove</button>
      </div>
    `;

    fileItems.appendChild(fileItem);
  });
}

function getFileIcon(filename) {
  const extension = filename.toLowerCase().split('.').pop();
  
  const iconMap = {
    'pdf': '📕',
    'doc': '📘',
    'docx': '📘',
    'txt': '📄',
    'xlsx': '📗',
    'xls': '📗',
    'csv': '📊',
    'ppt': '📙',
    'pptx': '📙',
    'jpg': '🖼️',
    'jpeg': '🖼️',
    'png': '🖼️',
    'gif': '🖼️'
  };
  
  return iconMap[extension] || '📎';
}

function removeFile(index) {
  const fileName = selectedFiles[index].name;
  selectedFiles.splice(index, 1);
  displayFiles();
  updateStats();
  showToast(`${fileName} removed`, 'info', 2000);
}

function updateStats() {
  fileCount.textContent = selectedFiles.length;
  const total = selectedFiles.reduce((sum, file) => sum + file.size, 0);
  totalSize.textContent = formatFileSize(total);
}

function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return (
    Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i]
  );
}

// ==================== UPLOAD HANDLER WITH INGESTION ====================
uploadBtn.addEventListener("click", async () => {
  if (selectedFiles.length === 0) {
    showToast('Please select at least one PDF file!', 'warning');
    return;
  }

  // Get user email from input field
  const emailInput = document.getElementById('userEmail');
  const userEmail = emailInput?.value?.trim();

  // Validate email
  if (!userEmail || !userEmail.includes('@')) {
    showToast('Please enter a valid email address!', 'error');
    emailInput?.focus();
    return;
  }

  // Disable button during upload
  uploadBtn.disabled = true;
  uploadBtn.textContent = '⏳ Uploading...';

  try {
    // ========== STEP 1: UPLOAD FILES ==========
    showToast(`Uploading ${selectedFiles.length} file(s)...`, 'info', 2000);
    
    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append('files', file);
    });

    const uploadUrl = `/upload/multiple?email=${encodeURIComponent(userEmail)}`;

    console.log(`📤 Step 1: Uploading ${selectedFiles.length} PDF file(s)...`);

    const uploadResponse = await fetch(uploadUrl, {
      method: 'POST',
      body: formData
    });

    if (!uploadResponse.ok) {
      throw new Error(`Upload failed: ${uploadResponse.status} ${uploadResponse.statusText}`);
    }

    const uploadResult = await uploadResponse.json();
    console.log('✅ Upload response:', uploadResult);

    if (uploadResult.status !== 'ok') {
      throw new Error(uploadResult.message || 'Upload failed');
    }

    // Show upload success
    const successfulUploads = uploadResult.uploaded_files.filter(f => !f.error);
    const failedUploads = uploadResult.uploaded_files.filter(f => f.error);
    
    if (successfulUploads.length > 0) {
      showToast(`✅ ${successfulUploads.length} file(s) uploaded successfully!`, 'success', 3000);
    }
    
    if (failedUploads.length > 0) {
      showToast(`⚠️ ${failedUploads.length} file(s) failed to upload`, 'warning', 3000);
    }

    // ========== STEP 2: EXTRACT FILE PATHS ==========
    const filePaths = [];
    successfulUploads.forEach((file) => {
      if (file.blob_path) {
        filePaths.push(file.blob_path);
      }
    });

    console.log(`📁 Extracted ${filePaths.length} file paths for ingestion`);

    // ========== STEP 3: INGEST FILES ==========
    let ingestionResult = null;
    if (filePaths.length > 0) {
      uploadBtn.textContent = '🔄 Processing files...';
      showToast(`Processing ${filePaths.length} file(s)...`, 'info', 2000);
      
      console.log(`📥 Step 2: Starting ingestion for ${filePaths.length} file(s)...`);

      const ingestionResponse = await fetch('/ingest/files', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          email: uploadResult.email,
          file_paths: filePaths
        })
      });

      if (!ingestionResponse.ok) {
        console.warn(`⚠️ Ingestion returned status ${ingestionResponse.status}`);
      }

      ingestionResult = await ingestionResponse.json();
      console.log('✅ Ingestion response:', ingestionResult);

      // Show ingestion success
      if (ingestionResult.status === 'ok') {
        showToast(`✅ ${ingestionResult.successful_ingestions} file(s) processed successfully!`, 'success', 4000);
      } else {
        showToast(`⚠️ Ingestion completed with errors`, 'warning', 3000);
      }
    }

    // ========== STEP 4: BUILD DETAILED SUMMARY ==========
    let summaryMessage = `✅ File Upload & Processing Complete!\n\n`;
    summaryMessage += `📊 Upload Summary:\n`;
    summaryMessage += `━━━━━━━━━━━━━━━━━━━━━━\n`;
    summaryMessage += `Total Files: ${uploadResult.total_files}\n`;
    summaryMessage += `Successful Uploads: ${successfulUploads.length}\n`;
    summaryMessage += `Failed Uploads: ${failedUploads.length}\n`;
    summaryMessage += `Total Chunks: ${uploadResult.total_chunks}\n`;
    summaryMessage += `Email: ${uploadResult.email}\n\n`;

    // Show uploaded files
    summaryMessage += '📁 Uploaded Files:\n';
    summaryMessage += `━━━━━━━━━━━━━━━━━━━━━━\n`;
    uploadResult.uploaded_files.forEach((file, index) => {
      if (file.error) {
        summaryMessage += `${index + 1}. ❌ ${file.filename}\n   Error: ${file.error}\n\n`;
      } else {
        summaryMessage += `${index + 1}. ✅ ${file.filename}\n`;
        summaryMessage += `   Chunks: ${file.chunks}\n`;
        summaryMessage += `   Path: ${file.blob_path}\n\n`;
      }
    });

    // Show ingestion results if available
    if (ingestionResult && ingestionResult.ingestion_results) {
      summaryMessage += '\n🔄 Ingestion Results:\n';
      summaryMessage += `━━━━━━━━━━━━━━━━━━━━━━\n`;
      summaryMessage += `Successful: ${ingestionResult.successful_ingestions}\n`;
      summaryMessage += `Failed: ${ingestionResult.failed_ingestions}\n\n`;

      ingestionResult.ingestion_results.forEach((result, index) => {
        const fileName = result.file_path ? result.file_path.split('/').pop() : 'Unknown';
        if (result.status === 'success') {
          summaryMessage += `${index + 1}. ✅ ${fileName}\n`;
          summaryMessage += `   ${result.message}\n\n`;
        } else {
          summaryMessage += `${index + 1}. ❌ ${fileName}\n`;
          summaryMessage += `   Error: ${result.error}\n\n`;
        }
      });
    }

    // Show detailed summary in alert
    alert(summaryMessage);

    // Clear all files after successful processing
    selectedFiles = [];
    displayFiles();
    updateStats();
    fileInput.value = '';

    console.log('✅ All operations completed successfully');

    // Final success toast
    showToast('🎉 All operations completed!', 'success', 3000);

  } catch (error) {
    console.error('❌ Error:', error);
    showToast(`Error: ${error.message}`, 'error', 5000);
    alert(`❌ Error:\n\n${error.message}\n\nPlease check the console for more details.`);
  } finally {
    // Re-enable button
    uploadBtn.disabled = false;
    uploadBtn.textContent = '🚀 Upload All Files';
  }
});