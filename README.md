One of the most challenging problems I solved recently was optimizing a multi-model deep learning pipeline for my project AI-Powered Crop Disease Prediction System.Initially, the models (ResNet101, ConvNeXtBase, DenseNet121) were computationally heavy, resulting in slow inference times and inconsistent real-time predictions.
After profiling the system, I discovered bottlenecks in image preprocessing, model loading, and memory management.To address this, I redesigned the preprocessing workflow, applied selective layer unfreezing, used TensorFlow’s prefetch and AUTOTUNE for faster data flow, and implemented a dynamic model registry so only the selected model loads into memory.
These optimizations reduced memory usage significantly and improved prediction speed to 2–3 seconds per image, making the system practical for real-world agricultural use.
This experience strengthened my ability to engineer AI systems that balance accuracy, performance, and scalability in real deployment scenarios.

Step1: Use Anaconda Prompt > create an environment to run the project
Step2: commands: change path using cd<"frontend file location"> : streamlit run app.py
Step3: Enter username and password in the frontend to see the predictions
