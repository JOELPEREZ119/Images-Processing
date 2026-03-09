# Images-Processing
This repository contains the implementation and performance analysis of three fundamental image processing filters: Gaussian, Sobel, and Median. The project compares three different computational approaches to evaluate execution efficiency.

**🚀 Overview**
The goal of this project is to apply image filters to a grayscale image and measure the performance gains achieved through various optimization techniques:

Pure Python: Baseline implementation using standard loops.

NumPy: Optimized version using vectorized matrix operations.

Cython-style: High-performance approach using contiguous memory access and pre-allocated buffers.

**🛠️ Project Structure**
main.py: The orchestrator that runs all tests, measures timing, and generates the comparison table.

filters_pure_python.py: Implementation using native Python (no external libraries).

filters_numpy.py: Implementation using NumPy's vectorization capabilities.

filters_cython.py: Implementation simulating low-level C optimizations (contiguous arrays, integer arithmetic).


requirements.txt: List of necessary Python dependencies.

/output: Directory where filtered images and results are saved.

💻 Installation & Usage
1. Clone the repository
2. Install dependencies
It is recommended to use a virtual environment.

3. Run the analysis
You can run the project using the default synthetic test image:

Or provide your own image:

📊 Results
The script generates a performance table in the terminal and saves all filtered versions (Gaussian, Sobel, and Median) in the output/ folder.

Execution Example:
(Note: Results may vary based on hardware specifications.)

🎓 Academic Info
Students: 
- Joel Alejandro Perez Yupit (2309182)
- Josué Octavio Chan Caballero (2309048)
- Gael Alberto Lara Peña (2309133)


University: Universidad Politécnica de Yucatán (UPY)


Unit: Image Processing - Unit 2
