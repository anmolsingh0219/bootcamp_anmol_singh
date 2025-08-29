# NYU Financial Risk Engineering Bootcamp - Homework Portfolio

## Project Overview
This repository contains the complete homework assignments from the NYU Financial Risk Engineering Bootcamp, culminating in an end-to-end Options Pricing Model project that demonstrates the full Applied Financial Engineering Lifecycle.

## Project Structure

```
homework/
├── README.md                                    # This file
├── stage01_problem-framing-and-scoping/         # Problem definition and scope
├── stage02_tooling-setup_slides-outline/        # Environment and tool setup
├── stage03_python-fundamentals/                 # Core Python skills
├── stage04_data-acquisition-and-ingestion/      # Data sourcing and loading
├── stage05_data-storage/                        # Data organization and storage
├── stage06_data-preprocessing/                  # Data cleaning and validation
├── stage07_outliers-risk-assumptions/           # Outlier detection and risk analysis
├── stage08_exploratory-data-analysis/           # Data exploration and visualization
├── stage09_feature_engineering/                 # Feature creation and selection
├── stage10a_modelling-linear-regression/        # Linear regression modeling
├── stage10b_modeling-time-series-and-classification/ # Advanced modeling techniques
├── stage11_evalution-and-risk-communication/    # Model evaluation and uncertainty
├── stage12-results-reporting-delivery-design-stakeholder-communication_homework-starter/ # Stakeholder communication
├── stage13_productization/                      # Production deployment
├── handoff/                                     # Deployment and monitoring (stage14)
├── stage15_orchestration-system-design/         # Pipeline orchestration
└── stage16_lifecycle-review/                    # Final reflection and documentation
```

## Applied Financial Engineering Lifecycle Mapping

### Phase 1: Foundation (Stages 1-3)
- **Problem Framing**: Defined options pricing prediction problem with clear success metrics
- **Tooling Setup**: Configured Python environment, Jupyter notebooks, and development tools
- **Python Fundamentals**: Applied core data science libraries and programming patterns

### Phase 2: Data Engineering (Stages 4-6)
- **Data Acquisition**: Structured approach to options market data ingestion
- **Data Storage**: Organized data with proper versioning and directory structure
- **Data Preprocessing**: Implemented data cleaning, validation, and quality checks

### Phase 3: Analysis & Modeling (Stages 7-10)
- **Outlier Analysis**: Identified and handled extreme values in financial data
- **Exploratory Data Analysis**: Discovered patterns in options pricing data
- **Feature Engineering**: Created domain-specific features (moneyness, volatility-time)
- **Modeling**: Built linear regression model with performance evaluation

### Phase 4: Evaluation & Communication (Stages 11-12)
- **Risk Communication**: Bootstrap analysis and confidence interval estimation
- **Results Reporting**: Stakeholder-friendly documentation and visualization

### Phase 5: Production & Operations (Stages 13-15)
- **Productization**: Flask API, Streamlit dashboard, and deployment packaging
- **Deployment & Monitoring**: Production readiness with monitoring strategy
- **Orchestration**: Pipeline design with automation and dependency management

### Phase 6: Reflection & Documentation (Stage 16)
- **Lifecycle Review**: Comprehensive documentation and lessons learned

## Key Project Outcomes

### Technical Achievements
- **End-to-End Pipeline**: Complete data science workflow from ingestion to deployment
- **Production Deployment**: Working Flask API and Streamlit dashboard
- **Comprehensive Testing**: Bootstrap validation and sensitivity analysis
- **Documentation**: Clear stakeholder communication and technical documentation

### Business Results
- **Model Performance**: Linear regression with MAE $58.8 (best case) and R² 15.6%
- **Risk Assessment**: Identified model limitations and deployment recommendations
- **Stakeholder Value**: Translated technical findings into business decisions
- **Production Readiness**: Deployment strategy with monitoring and alerting

### Learning Outcomes
- **Technical Skills**: Python, ML, API development, data visualization
- **Business Acumen**: Financial domain knowledge, stakeholder communication
- **Engineering Practices**: Version control, testing, documentation, deployment
- **Risk Management**: Model validation, uncertainty quantification, monitoring

## Repository Organization

Each stage directory contains:
- **notebooks/**: Jupyter notebooks with analysis and development work
- **data/**: Input data and processed outputs (where applicable)
- **src/**: Reusable Python modules and utilities
- **README.md**: Stage-specific documentation and instructions

## Requirements and Dependencies

### Core Requirements
- Python 3.8+
- Jupyter Notebook
- Key Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

### Production Requirements (Stage 13)
- Flask 3.0.0
- Streamlit 1.32.0
- See `stage13_productization/requirements.txt` for complete list

### Environment Setup
```bash
# Create environment for specific stages
python -m venv stage_env
source stage_env/bin/activate  # Linux/Mac
# or
stage_env\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt
```

## How to Use This Repository

### For Learning
1. Start with `stage01_problem-framing-and-scoping/` for project context
2. Follow stages sequentially to understand the full lifecycle
3. Review `stage16_lifecycle-review/` for comprehensive reflection

### For Replication
1. Clone the repository
2. Set up Python environment with requirements
3. Run notebooks in sequential order
4. Deploy using Stage 13 production components

### For Portfolio Review
- **Executive Summary**: See `stage12-results-reporting.../deliverables/final_report.md`
- **Technical Details**: Review individual stage notebooks
- **Production Demo**: Use `stage13_productization/` Flask API and Streamlit app
- **Complete Lifecycle**: See `stage16_lifecycle-review/completed_framework_guide.md`

## Contact and Support

This repository represents academic coursework completed as part of the NYU Financial Risk Engineering Bootcamp. For questions about specific implementations or methodologies, refer to the documentation within each stage directory.

## Final Notes

This project demonstrates the complete Applied Financial Engineering Lifecycle, from problem identification through production deployment. While the Options Pricing Model has limitations (R² = 15.6%), the comprehensive approach showcases industry-standard practices for model development, validation, and deployment in financial contexts.
