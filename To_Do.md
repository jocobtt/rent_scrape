# Tokyo Rent Predictor - Enhanced To-Do List 🚀

## 🏗️ Infrastructure & DevOps (High Priority)
- [ ] **Fix Terraform Cloud Run deployment issues**
  - [X] Debug container startup error (PORT=8000 environment variable issue)
  - [X] Update Terraform to use new project structure (models/ directory)
  - [X] Add proper health checks and startup probes
  - [X] Configure proper resource limits and auto-scaling

## 🤖 ML Operations & Monitoring
- [X] **Model Monitoring & Drift Detection**
  - [X] Integrate Evidently AI for model drift monitoring
  - [X] Set up automated alerts for performance degradation - if this can be done with evidently then sure otherwise lets not worry about it..
  - [X] Implement A/B testing framework for challenger vs passed models

- [X] **MLflow Integration Enhancement**
  - [X] Set up MLflow tracking server (Cloud SQL + GCS backend)
  - [X] Implement proper experiment tracking for all models
  - [X] Add model registry with versioning and stage transitions
  - [X] Create automated model promotion pipeline

## 🧠 Model Improvements
- [X] **PyTorch Model Enhancement**
  - [X] Complete neural network implementation for rent prediction - test
  - [X] Add proper model validation and hyperparameter tuning
  - [X] Implement deep learning for complex feature interactions
  - [X] Compare performance with existing models

- [ ] **Model Performance**
  - [X] Implement cross-validation and proper model evaluation
  - [X] Add ensemble methods combining two ml based models
  - [X] Create model explainability features (SHAP, LIME)
  - [ ] Implement online learning for real-time model updates?

## 📊 Data Pipeline & Quality
- [ ] **Data Pipeline Automation**
  - [ ] Set up scheduled data scraping (monthly)
  - [ ] Implement data quality checks and validation - great expectations like? 
  - [X] Add data versioning and lineage tracking 

- [ ] **Data Enhancement**
  - [ ] Implement geocoding and location features
  - [ ] Add economic indicators (inflation, interest rates)?
  - [ ] Include neighborhood characteristics data?
  - [ ] Make sure that I can still scrape the websites reliably + better data managment

## 🌐 Production Readiness
- [X] **API Enhancements**
  - [X] add rate limiting
  - [X] Add comprehensive logging and metrics
  - [X] Create API documentation with examples

## 🔒 Security & Compliance
- [ ] **Security Hardening**
  - [X] Add input validation and sanitization
  - [ ] JWT auth
  - [X] Set up vulnerability scanning - just do github actions 

## 📚 Documentation & Best Practices
- [ ] **Documentation**
  - [ ] Create comprehensive architecture documentation
  - [X] Add API documentation with OpenAPI/Swagger
  - [ ] Write deployment and operation runbooks

- [ ] **Code Quality**
  - [ ] Add comprehensive test coverage (>80%)
  - [ ] Implement code quality gates (linting, formatting)
  - [ ] Add pre-commit hooks
  - [ ] Create development environment setup guide - docker compose reference in readme is good enough 

## 🎯 Resume Enhancement Features
- [ ] **Showcase Modern ML Practices**
  - [ ] Create automated model lifecycle management
  - [ ] Demonstrate cost optimization strategies? 

## 🔄 Immediate Next Steps (This Week)
1. Complete GitHub Actions CI/CD pipeline  
2. Add comprehensive testing suite
3. Update all documentation