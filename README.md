### **README for Image Similarity Analyzer**

---

# **Image Similarity Analyzer**

### **Description**
This project analyzes images to identify the most similar ones based on advanced techniques like:
- **SSIM** (Structural Similarity Index).
- **ORB** (Oriented FAST and Rotated BRIEF keypoints matching).
- **SIFT** (Scale-Invariant Feature Transform).

The program generates comprehensive reports in multiple formats **(PDF, CSV)** and visualizes results through detailed plots.  
> **Created by Scienceveli** 🌟

---

### **Features**
- **Advanced Image Analysis**: Compares images using modern algorithms for accuracy.
- **Customizable Weights**: Adjust SSIM, ORB, and SIFT weights via a flexible `config.json` file.
- **Professional Reporting**: Generates detailed PDF reports with included images.
- **Quality Filtering**: Filters out low-resolution images automatically.
- **High Performance**: Multithreading support for faster processing.

---

### **Requirements**
- Python 3.7 or higher.
- Required Python libraries:
  ```bash
  pip install numpy opencv-python scikit-image pandas fpdf matplotlib
  ```

---

### **How to Use**
1. **Prepare Images**:  
   Place your images in a directory and specify its path in the `config.json` file.

2. **Configure Settings**:  
   Use a `config.json` file to define your preferences. Example:
   ```json
   {
       "image_directory": "path/to/your/images",
       "num_images": 100,
       "min_image_resolution": [720, 720],
       "weights": {
           "ssim": 0.4,
           "orb": 0.3,
           "sift": 0.3
       }
   }
   ```

3. **Run the Script**:  
   Execute the script:
   ```bash
   python script_name.py
   ```

4. **Outputs**:  
   - **CSV File**: Lists the top similar images.
   - **PDF Report**: Includes selected images.
   - **Plot**: A graphical representation of image similarity scores.

---

### **Example Outputs**
#### PDF Report  
A professional report with images and detailed information.  

#### CSV File  
A structured file listing the top 100 images.

#### Similarity Plot  
A bar chart showing similarity scores.

---

### **Customization**
- Modify the `config.json` file to:
  - Change the number of images (`num_images`).
  - Adjust similarity weights (`weights`).
  - Set a minimum image resolution (`min_image_resolution`).

---

### **Future Enhancements**
- Support for additional similarity algorithms.
- Web interface for interactive analysis.
- Integration with cloud storage for large-scale image datasets.

---

> **Developed and maintained by Scienceveli.** 🌍  
