# spam-classification-mlops-pipeline
A robust and end-to-end spam classification pipeline using MLOps best practices: Git, GitHub, DVC, custom logging, exception handling, modular pipeline automation, experiment tracking with DVC Live, and AWS S3 integration for scalable data versioning.

## 🚀 Key Features

- ✅ **Version Control**: All code and changes are tracked using Git and GitHub.
- 🧱 **Modular Architecture**: 
  - `data_ingestion`
  - `data_preprocessing`
  - `feature_engineering`
  - `model_building`
  - `model_evaluation`
- 🪵 **Custom Logging & Exception Handling**: For better debugging and traceability.
- 🛠️ **Pipeline Automation**: DVC (`dvc.yaml`) is used to automate the end-to-end workflow.
- ⚙️ **Parameterization**: Easily change hyperparameters and config values using `params.yaml`, Change parameters in one place and apply everywhere.
- 📊 **Experiment Tracking**: DVCLive is used to log and store metrics for every experiment. Track and compare all experiments.
- 📈 **Experiment Visualization**: DVC extension (In VS Code) allows experiment comparison and visualization.
